"""
ChunkFormer: Masked Chunking Conformer For Long-Form Speech Transcription

A PyTorch implementation of ChunkFormer for automatic speech recognition (ASR)
that efficiently handles long-form audio transcription on low-memory GPUs.
"""

__version__ = "0.1.1"
__author__ = "khanld"
__email__ = "khanhld218@gmail.com"

from chunkformer.chunkformer_model import ChunkFormerConfig, ChunkFormerModel

__all__ = ["ChunkFormerModel", "ChunkFormerConfig", "__version__"]
