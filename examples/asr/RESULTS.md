
# ChunkFormer Model Results

This document contains evaluation results for ChunkFormer models across different architectures and configurations.
- **Context Configuration Format**: (chunk_size, left_context, right_context). (-1, -1, -1) means full context processing.

---

## CTC Models

### ChunkFormer-CTC-Small-Libri-100h

#### Model Configuration
- **Training Dataset**: LibriSpeech 100h
- **Configuration File**: `examples/asr/ctc/conf/v0.yaml`
- **Checkpoint**: [![Hugging Face](https://img.shields.io/badge/HuggingFace-chunkformer--ctc--small--libri--100h-orange?logo=huggingface)](https://huggingface.co/khanhld/chunkformer-ctc-small-libri-100h)
- **Hardware**: 2 GPUs
- **Search Algorithm**: Greedy Search

#### Important Notes
- **Purpose**: These results are for pipeline validation and functionality verification
- **Optimization Status**: No hyperparameter tuning has been performed; results represent baseline performance

### Results (WER)

#### LibriSpeech Test Sets

| Test Set       | (-1, -1, -1)   | (64, 128, 128) | (128, 128, 128) | (128, 256, 256) | (256, 128, 128) |
|:---------------|:--------------:|:--------------:|:---------------:|:---------------:|:---------------:|
| **test-clean** |    **8.75**    |      8.87      |      8.87       |      8.80       |      8.77       |
| **test-other** |   **25.55**    |     25.55      |     25.59       |     25.57       |     25.56       |

---

## RNN-T Models
