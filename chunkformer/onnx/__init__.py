"""ONNX export and ONNX Runtime inference for ChunkFormer.

This subpackage provides:

- :mod:`chunkformer.onnx.wrappers`: thin ``nn.Module`` wrappers that expose
  ONNX-clean (tensor-only) forward signatures for the encoder (streaming and
  non-streaming), the CTC head, and the RNN-T predictor/joint networks.
- :mod:`chunkformer.onnx.runtime`: an :class:`OnnxAsrModel` that loads the
  exported graphs with ONNX Runtime and performs host-side CTC / RNN-T greedy
  decoding (including the cache-aware streaming chunk loop).

The actual export CLI lives in ``tools/export_onnx.py``.
"""

from chunkformer.onnx.wrappers import (
    CtcOnnx,
    EncoderChunkOnnx,
    EncoderFullOnnx,
    JointOnnx,
    PredictorStepOnnx,
)

__all__ = [
    "CtcOnnx",
    "EncoderChunkOnnx",
    "EncoderFullOnnx",
    "JointOnnx",
    "PredictorStepOnnx",
]
