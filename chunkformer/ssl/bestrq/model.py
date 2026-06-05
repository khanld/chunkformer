# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple

import torch
import torch.nn as nn

from chunkformer.ssl.modules.loss import MLMLoss
from chunkformer.ssl.modules.quantizer import RandomProjectionVectorQuantizer
from chunkformer.ssl.modules.utils import index_put
from chunkformer.utils.mask import compute_mask_indices, make_pad_mask


class BestRQ(torch.nn.Module):
    """BEST-RQ pretraining wrapper around a ChunkFormer encoder.

    Masks the input fbank features, predicts the random-projection-quantizer
    codebook ids of the masked frames, and optimises an MLM (NLL) objective.
    The quantizer is frozen (random projection + random codebook), following the
    BEST-RQ recipe.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_embed_dim: int = 768,
        final_dim: int = 0,
        logit_temp: float = 0.1,
        latent_vars: int = 8192,
        latent_groups: int = 1,
        latent_dim: int = 16,
        mask_length: int = 10,
        mask_prob: float = 0.65,
        mask_selection: str = "static",
        mask_other: float = 0,
        no_mask_overlap: bool = False,
        mask_min_space: int = 1,
        require_same_masks: bool = True,
        mask_dropout: float = 0.0,
        mask_channel_length: int = 10,
        mask_channel_prob: float = 0.0,
        mask_channel_before: bool = False,
        mask_channel_selection: str = "static",
        mask_channel_other: float = 0,
        no_mask_channel_overlap: bool = False,
        mask_channel_min_space: int = 1,
        dist_fn: str = "l2",
        freeze_quantizer: bool = True,
    ):
        super().__init__()
        self.encoder = encoder

        self.mask_dropout = mask_dropout
        self.require_same_masks = require_same_masks
        self.mask_prob = mask_prob
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.mask_length = mask_length
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space

        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_before = mask_channel_before
        self.mask_channel_selection = mask_channel_selection
        self.mask_channel_other = mask_channel_other
        self.mask_channel_length = mask_channel_length
        self.no_mask_channel_overlap = no_mask_channel_overlap
        self.mask_channel_min_space = mask_channel_min_space

        self.logit_temp = logit_temp

        final_dim = final_dim if final_dim > 0 else encoder_embed_dim

        self.quantizer = RandomProjectionVectorQuantizer(
            feat_in=self.encoder.embed._feat_in * self.encoder.embed.reverse_calc_length(1),
            code_dim=latent_dim,
            num_classes=latent_vars,
            num_books=latent_groups,
            dist_fn=dist_fn,
            freeze=freeze_quantizer,
        )

        self.mask_emb = nn.Parameter(torch.FloatTensor(self.encoder.embed._feat_in).uniform_())
        self.encoder = encoder

        self.final_proj = nn.Linear(encoder_embed_dim, latent_groups * latent_vars)

        # loss
        self.criterion = MLMLoss()

    def apply_mask(
        self,
        x,
        padding_mask,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        B, T, C = x.shape

        if self.mask_channel_prob > 0 and self.mask_channel_before:
            mask_channel_indices = compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices).to(x.device).unsqueeze(1).expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        if self.mask_prob > 0:
            if mask_indices is None:
                mask_indices = compute_mask_indices(
                    (B, T),
                    padding_mask,
                    self.mask_prob,
                    self.mask_length,
                    self.mask_selection,
                    self.mask_other,
                    min_masks=2,
                    no_overlap=self.no_mask_overlap,
                    min_space=self.mask_min_space,
                    require_same_masks=self.require_same_masks,
                    mask_dropout=self.mask_dropout,
                )
                mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x = index_put(x, mask_indices, self.mask_emb.to(x.dtype))
        else:
            mask_indices = None

        if self.mask_channel_prob > 0 and not self.mask_channel_before:
            if mask_channel_indices is None:
                mask_channel_indices = compute_mask_indices(
                    (B, C),
                    None,
                    self.mask_channel_prob,
                    self.mask_channel_length,
                    self.mask_channel_selection,
                    self.mask_channel_other,
                    no_overlap=self.no_mask_channel_overlap,
                    min_space=self.mask_channel_min_space,
                )
                mask_channel_indices = (
                    torch.from_numpy(mask_channel_indices)
                    .to(x.device)
                    .unsqueeze(1)
                    .expand(-1, T, -1)
                )
            x = index_put(x, mask_channel_indices, 0)

        return x, mask_indices

    def feature_extractor(
        self,
        xs: torch.Tensor,
        xs_lens: torch.Tensor,
        chunk_size: int = 0,
        left_context_size: int = 0,
        right_context_size: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)  # (B, T)

        xs_norm = xs.clone()
        # set the dropout to 0.0 during feature extraction
        xs, pos_emb, masks = self.encoder.embed(
            xs,
            masks,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
        )
        return xs, pos_emb, masks, xs_norm

    def stacking(self, xs):
        xs = self.encoder.embed.stacking(xs)
        return xs

    def forward(
        self,
        batch: dict,
        device: torch.device,
        mask_indices=None,
        mask_channel_indices=None,
    ):
        xs = batch["feats"].to(device)
        xs_lens = batch["feats_lengths"].to(device)

        if self.encoder.global_cmvn is not None:
            xs = self.encoder.global_cmvn(xs)
        unmask_xs = xs.clone()

        padding_mask = make_pad_mask(xs_lens, xs.size(1))  # (B, T)
        xs, mask_indices = self.apply_mask(
            xs,
            padding_mask,
            mask_indices=mask_indices,
            mask_channel_indices=mask_channel_indices,
        )

        chunk_size = 0
        left_context_size = 0
        right_context_size = 0

        x, pos_emb, x_mask, xs_norm = self.feature_extractor(
            xs, xs_lens, chunk_size, left_context_size, right_context_size
        )

        unmasked_features = self.stacking(unmask_xs)
        unmasked_features = unmasked_features.reshape(
            unmasked_features.size(0), unmasked_features.size(1), -1
        )
        # B, T, L
        mask_indices = self.stacking(mask_indices)
        # B, T
        mask_indices = mask_indices.float().mean(-1)
        # B, T
        mask_indices = mask_indices >= 0.8

        padding_mask = ~x_mask

        x, _, _ = self.encoder.forward_layers(
            x,
            x_mask,
            pos_emb,
            x_mask,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
        )
        if self.encoder.normalize_before and self.encoder.final_norm:
            x = self.encoder.after_norm(x)

        logit = x[mask_indices]
        logit = self.final_proj(logit)
        logit = logit.reshape(-1, self.quantizer.num_books, self.quantizer.num_classes)
        logit = logit.log_softmax(-1)

        _, y, code_perplexity, prob_perplexity = self.quantizer(unmasked_features, mask_indices)

        y = y[mask_indices]

        y = y.reshape(-1)
        logit = logit.reshape(-1, self.quantizer.num_classes)
        pred = logit.argmax(-1)

        corr = (pred == y).to(torch.float32).mean().item()

        loss = self.criterion(logit, y)

        mask_percentile = (mask_indices.sum(-1).float() / x_mask.float().sum(-1)).mean().item()

        logging_output = {
            "loss": loss,
            "corr": corr,
            "ntokens": mask_indices.sum(),
            "mask_percentile": mask_percentile,
            "nsentences": xs.size(0),
            "code_perplexity": code_perplexity,
            "prob_perplexity": prob_perplexity,
        }
        return logging_output
