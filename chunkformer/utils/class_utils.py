#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
import torch
from torch.nn import BatchNorm1d, LayerNorm

from chunkformer.modules.attention import MultiHeadedAttention, MultiHeadedCrossAttention
from chunkformer.modules.embedding import PositionalEncoding
from chunkformer.modules.norm import RMSNorm
from chunkformer.modules.positionwise_feed_forward import PositionwiseFeedForward
from chunkformer.modules.swish import Swish

WENET_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}

WENET_RNN_CLASSES = {
    "rnn": torch.nn.RNN,
    "lstm": torch.nn.LSTM,
    "gru": torch.nn.GRU,
}


WENET_NORM_CLASSES = {"layer_norm": LayerNorm, "batch_norm": BatchNorm1d, "rms_norm": RMSNorm}

WENET_ATTENTION_CLASSES = {"selfattn": MultiHeadedAttention, "crossattn": MultiHeadedCrossAttention}

WENET_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
}

WENET_MLP_CLASSES = {
    "position_wise_feed_forward": PositionwiseFeedForward,
}
