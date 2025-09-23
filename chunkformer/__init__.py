"""
ChunkFormer: Masked Chunking Conformer For Long-Form Speech Transcription

A PyTorch implementation of ChunkFormer for automatic speech recognition (ASR)
that efficiently handles long-form audio transcription on low-memory GPUs.
"""

__version__ = "0.1.0"
__author__ = "khanld"
__email__ = "khanhld218@gmail.com"

from .chunkformer_model import ChunkFormerModel, ChunkFormerConfig

__all__ = [
    "ChunkFormerModel",
    "ChunkFormerConfig", 
    "__version__"
]