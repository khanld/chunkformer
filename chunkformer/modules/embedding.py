"""Positonal Encoding Module."""

import math
from typing import Tuple, Union

import torch
import torch.nn.functional as F


class PositionalEncoding(torch.nn.Module):
    """Positional encoding.

    :param int d_model: embedding dim
    :param float dropout_rate: dropout rate
    :param int max_len: maximum input length

    PE(pos, 2i)   = sin(pos/(10000^(2i/dmodel)))
    PE(pos, 2i+1) = cos(pos/(10000^(2i/dmodel)))
    """

    def __init__(
        self, d_model: int, dropout_rate: float, max_len: int = 5000, reverse: bool = False
    ):
        """Construct an PositionalEncoding object."""
        super().__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.max_len = max_len

        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(
        self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
            offset (int, torch.tensor): position offset

        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
            torch.Tensor: for compatibility to RelPositionalEncoding
        """

        pos_emb = self.position_encoding(offset, x.size(1), False)
        x = x * self.xscale + pos_emb
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(
        self, offset: Union[int, torch.Tensor], size: int, apply_dropout: bool = True
    ) -> torch.Tensor:
        """For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        # How to subscript a Union type:
        #   https://github.com/pytorch/pytorch/issues/69434
        if isinstance(offset, int):
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        elif isinstance(offset, torch.Tensor) and offset.dim() == 0:  # scalar
            assert offset + size <= self.max_len
            pos_emb = self.pe[:, offset : offset + size]
        else:  # for batched streaming decoding on GPU
            assert torch.max(offset) + size <= self.max_len
            index = offset.unsqueeze(1) + torch.arange(0, size).to(offset.device)  # B X T
            flag = index > 0
            # remove negative offset
            index = index * flag
            pos_emb = F.embedding(index, self.pe[0])  # B X T X d_model

        if apply_dropout:
            pos_emb = self.dropout(pos_emb)
        return pos_emb


class RelPositionalEncodingWithRightContext(torch.nn.Module):
    """Relative positional encoding module.

    Args:
        d_model: Embedding dimension.
        dropout_rate: Dropout rate.
        max_len: Maximum input length.
    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000) -> None:
        """Construct an PositionalEncoding object."""
        super(RelPositionalEncodingWithRightContext, self).__init__()

        self.d_model = d_model
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe: torch.Tensor  # Will be initialized in extend_pe
        self.xscale = math.sqrt(self.d_model)
        self.max_len = max_len
        self.extend_pe(max_len)

    def extend_pe(self, size: int) -> None:
        """Reset the positional encodings."""
        # Suppose `i` means to the position of query vector and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(size, self.d_model)
        pe_negative = torch.zeros(size, self.d_model)
        position = torch.arange(0, size, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in "Transformer-XL: Attentive Language Models Beyond a
        # Fixed-Length Context"
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        self.pe = torch.cat([pe_positive, pe_negative], dim=1)

    def position_encoding(
        self,
        chunk_size: Union[int, torch.Tensor] = 0,
        left_context_size: Union[int, torch.Tensor] = 0,
        right_context_size: Union[int, torch.Tensor] = 0,
        apply_dropout: bool = False,
    ) -> torch.Tensor:
        if isinstance(left_context_size, int):
            assert left_context_size + chunk_size < self.max_len
            x_size_1 = chunk_size + left_context_size
            pos_emb = self.pe[
                :,
                self.pe.size(1) // 2
                - x_size_1
                + 1 : self.pe.size(1) // 2  # noqa E203
                + chunk_size
                + right_context_size,
            ]
        else:
            assert left_context_size + chunk_size < self.max_len
            x_size_1 = chunk_size + left_context_size
            pos_emb = self.pe[
                :,
                self.pe.size(1) // 2
                - x_size_1
                + 1 : self.pe.size(1) // 2  # noqa E203
                + chunk_size
                + right_context_size,
            ]

        return pos_emb

    def forward(
        self,
        x: torch.Tensor,
        chunk_size: Union[int, torch.Tensor] = 0,
        left_context_size: Union[int, torch.Tensor] = 0,
        right_context_size: Union[int, torch.Tensor] = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).
            offset (int): left context (in frames) used during streaming decoding.
                this is used only in real streaming decoding, in other circumstances,
                it MUST be 0.
            chunk_size (int): Chunk size for limited chunk context
            left_context_size (int): Left context size for limited chunk context
            right_context_size (int): Right context size for limited chunk context
        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).
            torch.Tensor: Encoded tensor (batch, 2*time-1, `*`).

        """
        x = x * self.xscale
        chunk_size = x.size(1) if chunk_size <= 0 else chunk_size
        pos_emb = self.position_encoding(
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
            apply_dropout=False,
        ).to(device=x.device, dtype=x.dtype)
        return self.dropout(x), self.dropout(pos_emb)
