"""
ChunkFormer: Masked Chunking Conformer For Long-Form Speech Transcription

A PyTorch implementation of ChunkFormer for automatic speech recognition (ASR)
that efficiently handles long-form audio transcription on low-memory GPUs.
"""

__version__ = "1.2.0"
__author__ = "khanld"
__email__ = "khanhld218@gmail.com"

from chunkformer.chunkformer_model import ChunkFormerModel

__all__ = ["ChunkFormerModel", "__version__"]
