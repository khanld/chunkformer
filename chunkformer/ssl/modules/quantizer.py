"""Random-projection quantizers used by ViP-VL pretraining.

ref: https://arxiv.org/pdf/2202.01855 (Self-Supervised Learning with
Random-Projection Quantizer for Speech Recognition).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.linalg import vector_norm


class RandomProjectionQuantizer(nn.Module):
    """Vector quantization using a projection and a randomly initialised codebook
    this is useful for models like ViP-VL for instance.

    The output is the indices of the closest code in the codebook for each
    time step of the input.

    ref: https://arxiv.org/pdf/2202.01855

    Arguments
    ---------
    input_dim: int
        Input dimension (channels).
    cb_dim: int
        Size of each code in the codebook.
    cb_vocab: int
        Number of codes in the codebook

    Example
    -------
    >>> quantiser = RandomProjectionQuantizer(16, 16, 32)
    >>> inputs = torch.rand(10, 12, 16)
    >>> output = quantiser(inputs)
    >>> output.shape
    torch.Size([10, 12])
    """

    def __init__(self, input_dim, cb_dim, cb_vocab):
        super().__init__()

        self.input_dim = input_dim
        self.cb_dim = cb_dim
        self.cb_vocab = cb_vocab

        # Section 3.1 "projection matrix A use Xavier initialization"
        P_init = torch.empty((input_dim, cb_dim))
        self.register_buffer("P", nn.init.xavier_uniform_(P_init))

        # normalize random matrix for codebook
        self.register_buffer("CB", F.normalize(torch.randn(cb_vocab, cb_dim)))

    def forward(self, x):
        """Forward the latent vector to obtain a quantised output"""

        x = F.normalize(x @ self.P, dim=2)
        return vector_norm((self.CB.unsqueeze(1) - x.unsqueeze(1)), dim=-1).argmin(dim=1)


class RandomProjectionVectorQuantizer(nn.Module):
    DIST_FN_LIST = ["l2"]

    def __init__(
        self,
        feat_in: int,
        code_dim: int,
        num_classes: int,
        num_books: int,
        dist_fn: str = "l2",
        freeze: bool = True,
    ):
        """Vector quantization using random projection proposed in BEST-RQ paper:
        'Self-Supervised Learning with Random-Projection Quantizer for Speech Recognition'

         Args:
            feat_in: input feature dimension
            code_dim: dimension of the codebook features
            num_classes: number of classes
            num_books: number of codebooks
            dist_fn: distance function to use; only "l2" is supported
            freeze: whether to freeze the projection matrix
        """
        super().__init__()

        if dist_fn not in self.DIST_FN_LIST:
            raise ValueError(
                f"Unknown distance function {dist_fn}, must be one of {self.DIST_FN_LIST}"
            )

        self.feat_in = feat_in
        self.code_dim = code_dim
        self.num_classes = num_classes
        self.num_books = num_books
        self.dist_fn = dist_fn

        # (B, T, D) -> (B, T, num_books, code_dim)
        self.proj = nn.Linear(self.feat_in, self.num_books * self.code_dim, bias=False)
        torch.nn.init.xavier_normal_(self.proj.weight)

        # (num_books, num_classes, hid_dim)
        codebooks = torch.randn(self.num_books, self.num_classes, self.code_dim).double()
        torch.nn.init.normal_(codebooks, mean=0, std=1)
        codebooks = F.normalize(codebooks, dim=-1)
        self.codebooks = nn.Parameter(codebooks)
        if freeze:
            self.freeze()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, input_signal, mask_indices=None):
        """
        Args:
            input_signal: input features of shape (B, T, D) or (B, D, T)
        Returns:
            xq: quantized features of shape (B, T, D, N) or (B, D, T, N)
            xid: quantized tokens of shape (B, T, N)
        """

        B, T, _ = input_signal.size()

        # (B, T, D) -> (B, T, num_books*code_dim)
        x = self.proj(input_signal)

        # normalize each feature vector
        # (B, T, num_books*code_dim) -> (B, T, num_books, code_dim)
        x = F.normalize(x.view(B, T, self.num_books, self.code_dim), dim=-1)

        # get tokens (xid) of shape (B, T, num_books) using l2 distance
        # (B, T, num_books, code_dim) -> (B, T, num_books, code_dim, num_classes)
        xid = x.unsqueeze(-1) - self.codebooks.transpose(1, 2).unsqueeze(0).unsqueeze(0)
        # (B, T, num_books, num_classes)
        xid_soft = xid.norm(dim=-2)

        xid_hard = torch.zeros_like(xid_soft, device=xid.device)
        xid = xid_soft.argmin(dim=-1)

        # B, T, num_books, num_classes
        xid_hard.scatter_(dim=-1, index=xid.unsqueeze(-1), value=1.0)

        # xid2: (B, T, num_books) -> (B, T, num_books)
        xid2 = xid + self.num_classes * torch.arange(self.num_books, device=xid.device).unsqueeze(
            0
        ).unsqueeze(0)
        # xid2: (B, T, num_books) -> (B*num_books, T)
        xid2 = xid2.transpose(1, 2).contiguous().view(-1, T)

        # get quantized vector (xq) of shape (B, T, code_dim, num_books)
        # codebook: (num_books, num_classes, code_dim) -> (num_books*num_classes, code_dim)
        xq = F.embedding(xid2.view(-1), self.codebooks.view(-1, self.code_dim)).view(
            B, T, self.code_dim, self.num_books
        )

        # (B, T, num_books, num_classes) -> (B * T, num_books, num_classes)
        xid_hard = xid_hard.reshape(
            xid_hard.size(0) * xid_hard.size(1), self.num_books, self.num_classes
        )
        # (B * T, num_books, num_classes) -> (num_books, num_classes)
        xid_hard = torch.mean(xid_hard.float(), dim=0)
        code_perplexity = torch.exp(-torch.sum(xid_hard * torch.log(xid_hard + 1e-7), dim=-1)).sum()

        xid_soft = torch.softmax(
            xid_soft.view(xid_soft.size(0) * xid_soft.size(1), self.num_books, -1).float(), dim=-1
        ).mean(dim=0)

        prob_perplexity = torch.exp(-torch.sum(xid_soft * torch.log(xid_soft + 1e-7), dim=-1)).sum()
        return xq, xid, code_perplexity, prob_perplexity
