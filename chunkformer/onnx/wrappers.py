"""ONNX-clean ``nn.Module`` wrappers around ChunkFormer submodules.

Each wrapper exposes a ``forward`` that takes and returns only plain tensors
(no dicts, lists, ``Optional`` or python-int control flow that depends on the
data), which is the form ``torch.onnx.export`` handles reliably.

The streaming encoder bakes ``chunk_size`` / ``left_context_size`` /
``right_context_size`` as python ints at construction time (so the traced graph
is specialised for one streaming configuration) while keeping ``offset`` as a
tensor input so the warm-up masking for the first chunks is computed correctly
inside the graph.
"""

from typing import Tuple

import torch


def _disable_subsampling_conv_chunking(encoder: torch.nn.Module) -> None:
    """Force the simple ``self.conv(x)`` path in the subsampling module.

    The depthwise-striding subsampling has a data-dependent ``need_to_split``
    branch (guarded by ``subsampling_conv_chunking_factor``) used to work around
    a 2**31 element indexing limit for very large batches. Setting the factor to
    ``-1`` skips that branch entirely, which keeps the exported ONNX graph clean
    and is always correct for the (small) tensors used at inference time.
    """
    if hasattr(encoder, "embed") and hasattr(encoder.embed, "subsampling_conv_chunking_factor"):
        encoder.embed.subsampling_conv_chunking_factor = -1


class EncoderFullOnnx(torch.nn.Module):
    """Non-streaming encoder.

    Supports both full-context decoding (``chunk_size=0``) and ChunkFormer's
    masked limited-context decoding (``chunk_size>0`` with left/right context),
    which is the efficient long-form mode. The context sizes are baked in at
    construction; ``chunk_size=0`` means full context.

    Inputs:
        feats: (B, T, feat_dim) raw fbank features (CMVN is applied internally).
        feat_lens: (B,) int lengths in frames.
    Outputs:
        enc_out: (B, T', D) encoder output.
        enc_lens: (B,) int output lengths.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        chunk_size: int = 0,
        left_context_size: int = 0,
        right_context_size: int = 0,
    ):
        super().__init__()
        _disable_subsampling_conv_chunking(encoder)
        self.encoder = encoder
        self.chunk_size = int(chunk_size)
        self.left_context_size = int(left_context_size)
        self.right_context_size = int(right_context_size)

    def forward(
        self, feats: torch.Tensor, feat_lens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_out, masks = self.encoder.forward_encoder(  # type: ignore[operator]
            feats,
            feat_lens,
            chunk_size=self.chunk_size,
            left_context_size=self.left_context_size,
            right_context_size=self.right_context_size,
        )
        enc_lens = masks.squeeze(1).sum(1).to(torch.int64)
        return enc_out, enc_lens


class EncoderChunkOnnx(torch.nn.Module):
    """Cache-aware streaming encoder for a single chunk.

    ``chunk_size`` / ``left_context_size`` / ``right_context_size`` are baked in
    at construction. Inputs:
        chunk: (B, T_chunk, feat_dim) raw fbank features for one window of
            ``size = reverse_calc_length(chunk_size) + right_context_size * sub``
            frames.
        att_cache: (num_blocks, B, heads, left_context_size, 2*d_k).
        cnn_cache: (num_blocks, B, D, cnn_module_kernel//2).
        offset: scalar int64 tensor, number of already-emitted subsampled frames.
    Outputs:
        enc_out: (B, chunk_size(+right at tail), D).
        r_att_cache / r_cnn_cache: updated caches with the same shapes as inputs.
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        chunk_size: int,
        left_context_size: int,
        right_context_size: int,
    ):
        super().__init__()
        _disable_subsampling_conv_chunking(encoder)
        self.encoder = encoder
        self.chunk_size = int(chunk_size)
        self.left_context_size = int(left_context_size)
        self.right_context_size = int(right_context_size)

    def forward(
        self,
        chunk: torch.Tensor,
        att_cache: torch.Tensor,
        cnn_cache: torch.Tensor,
        offset: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_out, _, r_att_cache, r_cnn_cache = self.encoder.forward_chunk(  # type: ignore[operator]
            chunk,
            att_cache=att_cache,
            cnn_cache=cnn_cache,
            chunk_size=self.chunk_size,
            left_context_size=self.left_context_size,
            right_context_size=self.right_context_size,
            offset=offset,
        )
        return enc_out, r_att_cache, r_cnn_cache


class CtcOnnx(torch.nn.Module):
    """CTC head: encoder output -> log-probabilities.

    Inputs:
        enc_out: (B, T, D).
    Outputs:
        log_probs: (B, T, V).
    """

    def __init__(self, ctc: torch.nn.Module):
        super().__init__()
        self.ctc = ctc

    def forward(self, enc_out: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.ctc.log_softmax(enc_out)  # type: ignore[operator]
        return out


class PredictorStepOnnx(torch.nn.Module):
    """RNN-T predictor single step (LSTM with explicit state cache).

    Inputs:
        token: (B, 1) int64 previous token id.
        state_m: (num_layers, B, hidden) LSTM hidden state.
        state_c: (num_layers, B, hidden) LSTM cell state.
    Outputs:
        pred_out: (B, 1, P) predictor projection.
        new_m / new_c: updated LSTM states.
    """

    def __init__(self, predictor: torch.nn.Module):
        super().__init__()
        self.predictor = predictor

    def forward(
        self, token: torch.Tensor, state_m: torch.Tensor, state_c: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embed = self.predictor.dropout(self.predictor.embed(token))  # type: ignore[operator]
        states = (state_m, state_c)
        pred_out, (new_m, new_c) = self.predictor.rnn(embed, states)  # type: ignore[operator]
        pred_out = self.predictor.projection(pred_out)  # type: ignore[operator]
        return pred_out, new_m, new_c


class JointOnnx(torch.nn.Module):
    """RNN-T joint network.

    Inputs:
        enc_t: (B, 1, E) encoder output for one frame.
        pred: (B, 1, P) predictor output for one step.
    Outputs:
        logits: (B, 1, V) joint logits (no log_softmax; matches PyTorch joint).
    """

    def __init__(self, joint: torch.nn.Module):
        super().__init__()
        self.joint = joint

    def forward(self, enc_t: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.joint(enc_t, pred)  # type: ignore[operator]
        return out
