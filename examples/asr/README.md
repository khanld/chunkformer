
# ChunkFormer Model Results

This document contains evaluation results for ChunkFormer models across different architectures and configurations.
- **Context Configuration Format**: (chunk_size, left_context, right_context). (-1, -1, -1) means full context processing.

---

## CTC Models

### ChunkFormer-CTC-Small-Libri-100h

#### Model Configuration
- **Training Dataset**: LibriSpeech 100h
- **Configuration File**: `examples/asr/ctc/conf/chunkformer-ctc-small-libri-100h.yaml`
- **Checkpoint**: [![Hugging Face](https://img.shields.io/badge/HuggingFace-chunkformer--ctc--small--libri--100h-orange?logo=huggingface)](https://huggingface.co/khanhld/chunkformer-ctc-small-libri-100h)
- **Hardware**: 2 GPUs
- **Search Algorithm**: Greedy Search

#### Important Notes
- **Purpose**: These results are for pipeline validation and functionality verification
- **Optimization Status**: No hyperparameter tuning has been performed; results represent baseline performance

#### Results on LibriSpeech Test Sets (WER)

| Test Set       | (-1, -1, -1)   | (64, 128, 128) | (128, 128, 128) | (128, 256, 256) | (256, 128, 128) |
|:---------------|:--------------:|:--------------:|:---------------:|:---------------:|:---------------:|
| **test-clean** |    8.75    |      8.87      |      8.87       |      8.80       |      8.77       |
| **test-other** |   25.55    |     25.55      |     25.59       |     25.57       |     25.56       |


### ChunkFormer-CTC-Small-Libri-960h

#### Model Configuration
- **Training Dataset**: LibriSpeech 960h
- **Configuration File**: `examples/asr/ctc/conf/chunkformer-ctc-small-libri-960h.yaml`
- **Checkpoint**: [![Hugging Face](https://img.shields.io/badge/HuggingFace-chunkformer--ctc--small--libri--960h-orange?logo=huggingface)](https://huggingface.co/khanhld/chunkformer-ctc-small-libri-960h)
- **Hardware**: 4 GPUs
- **Search Algorithm**: Greedy Search

#### Results on LibriSpeech Test Sets (WER)

| Test Set | (-1, -1, -1) | (64, 128, 128) | (128, 128, 128) | (128, 256, 256) | (256, 128, 128) |
|:---------------|:--------------:|:--------------:|:---------------:|:---------------:|:---------------:|
| **test-clean** | 3.19 | 3.22 | 3.20 | 3.20 | 3.18 |
| **test-other** | 8.51 | 8.54 | 8.53 | 8.53 | 8.51 |

### Chunkformer-CTC-Small-Libri-960h-stream-dct
#### Model Configuration
- **Training Dataset**: LibriSpeech 960h
- **Configuration File**: `examples/asr/ctc/conf/chunkformer-ctc-small-libri-960h-stream-dct.yaml`
- **Checkpoint**: [![Hugging Face](https://img.shields.io/badge/HuggingFace-chunkformer--ctc--small--libri--960h--stream--dct-orange?logo=huggingface)](https://huggingface.co/khanhld/chunkformer-ctc-small-libri-960h-stream-dct)
- **Hardware**: 4 GPUs
- **Search Algorithm**: Greedy Search
- **Streaming Config**: Each frame corresponds to 80ms. Right context is not used. Format: (left_context, chunk_size)

#### Results on LibriSpeech Test Sets (WER)

| Test Set | (50, 8) | (60, 8) | (50, 6) | (60, 6) | (50, 4) | (60, 4) |
|:---------------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| **test-clean** | 4.56 | 4.54 | 4.70 | 4.67 | 4.91 | 4.87 |
| **test-other** | 12.24 | 12.28 | 12.55 | 12.54 | 13.00 | 12.97 |

---

## RNN-T Models

### ChunkFormer-RNNT-Small-Libri-100h

#### Model Configuration
- **Training Dataset**: LibriSpeech 100h
- **Configuration File**: `examples/asr/rnnt/conf/chunkformer-rnnt-small-libri-100h.yaml`
- **Checkpoint**: [![Hugging Face](https://img.shields.io/badge/HuggingFace-chunkformer--rnnt--small--libri--100h-orange?logo=huggingface)](https://huggingface.co/khanhld/chunkformer-rnnt-small-libri-100h)
- **Hardware**: 2 GPUs
- **Search Algorithm**: Greedy Search

#### Important Notes
- **Purpose**: These results are for pipeline validation and functionality verification
- **Optimization Status**: No hyperparameter tuning has been performed; results represent baseline performance

| Test Set | (-1, -1, -1) | (64, 128, 128) | (128, 128, 128) | (128, 256, 256) | (256, 128, 128) |
|:---------------|:--------------:|:--------------:|:---------------:|:---------------:|:---------------:|
| **test-clean** | 8.09 | 8.22 | 8.22 | 8.15 | 8.10 |
| **test-other** | 22.88 | 23.03 | 22.93 | 22.92 | 22.88 |
