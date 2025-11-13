# ChunkFormer: Masked Chunking Conformer For Long-Form Speech Transcription
---
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Paper](https://img.shields.io/badge/Paper-ICASSP%202025-green)](https://arxiv.org/abs/2502.14673)

This repository contains the implementation and supplementary materials for our ICASSP 2025 paper, **"ChunkFormer: Masked Chunking Conformer For Long-Form Speech Transcription"**. The paper has been fully accepted by the reviewers with scores: **4/4/4**.

[![Ranked #1: Speech Recognition on Common Voice Vi](https://img.shields.io/badge/Ranked%20%231%3A%20Speech%20Recognition%20on%20Common%20Voice%20Vi-%F0%9F%8F%86%20SOTA-blueviolet?style=for-the-badge&logo=paperswithcode&logoColor=white)](https://paperswithcode.com/sota/speech-recognition-on-common-voice-vi)
[![Ranked #1: Speech Recognition on VIVOS](https://img.shields.io/badge/Ranked%20%231%3A%20Speech%20Recognition%20on%20VIVOS-%F0%9F%8F%86%20SOTA-blueviolet?style=for-the-badge&logo=paperswithcode&logoColor=white)](https://paperswithcode.com/sota/speech-recognition-on-vivos)



https://github.com/user-attachments/assets/aba64174-f965-43f2-92a2-7391fb0dba5c


## Table of Contents
- [Introduction](#introduction)
- [Key Features](#key-features)
- [Installation](#installation)
  - [Install from PyPI (Recommended)](#option-1-install-from-pypi-recommended)
  - [Install from source](#option-2-install-from-source)
  - [Pretrained Models](#pretrained-models)
- [Usage](#usage)
  - [Feature Extraction](#feature-extraction)
  - [Python API Transcription](#python-api-transcription)
  - [Command Line Transcription](#command-line-transcription)
- [Training the Model](#training)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)

<a name = "introduction" ></a>
## Introduction
ChunkFormer is an ASR model designed for processing long audio inputs effectively on low-memory GPUs. It uses a **chunk-wise processing mechanism** with **relative right context** and employs the **Masked Batch technique** to minimize memory waste due to padding. The model is scalable, robust, and optimized for both streaming and non-streaming ASR scenarios.
![chunkformer_architecture](docs/chunkformer_architecture.png)

<a name = "key-features" ></a>
## Key Features
- **Transcribing Extremely Long Audio**: ChunkFormer can **transcribe audio recordings up to 16 hours** in length with results comparable to existing models. It is currently the first model capable of handling this duration.
- **Efficient Decoding on Low-Memory GPUs**: Chunkformer can **handle long-form transcription on GPUs with limited memory** without losing context or mismatching the training phase.
- **Masked Batching Technique**: ChunkFormer efficiently **removes the need for padding in batches with highly variable lengths**.  For instance, **decoding a batch containing audio clips of 1 hour and 1 second costs only 1 hour + 1 second of computational and memory usage, instead of 2 hours due to padding.**

| GPU Memory | Total Batch Duration (minutes) |
|---|---|
| 80GB | 980 |
| 24GB | 240 |

<a name = "installation" ></a>
## Installation

### Option 1: Install from PyPI (Recommended)
```bash
pip install chunkformer
```

### Option 2: Install from source
```bash
# Clone the repository
git clone https://github.com/your-username/chunkformer.git
cd chunkformer

# Install in development mode
pip install -e .
```

### Pretrained Models
| Language | Model |
|----------|-------|
| Vietnamese  | [![Hugging Face](https://img.shields.io/badge/HuggingFace-chunkformer--rnnt--large--vie-orange?logo=huggingface)](https://huggingface.co/khanhld/chunkformer-rnnt-large-vie) |
| Vietnamese  | [![Hugging Face](https://img.shields.io/badge/HuggingFace-chunkformer--ctc--large--vie-orange?logo=huggingface)](https://huggingface.co/khanhld/chunkformer-ctc-large-vie) |
| English   | [![Hugging Face](https://img.shields.io/badge/HuggingFace-chunkformer--ctc--large--en--libri--960h-orange?logo=huggingface)](https://huggingface.co/khanhld/chunkformer-large-en-libri-960h) |
<a name = "usage" ></a>
## Usage

### Feature Extraction
```python
from chunkformer import ChunkFormerModel
import torch

device = "cuda:0"

# Load a pre-trained model from Hugging Face or local directory
model = ChunkFormerModel.from_pretrained("khanhld/chunkformer-ctc-large-vie").to(device)
x, x_len = model._load_audio_and_extract_features("path/to/audio")  # x: (T, F), x_len: int
x = x.unsqueeze(0).to(device)
x_len = torch.tensor([x_len], device=device)

# Extract feature
feature, feature_len = model.encode(
    xs=x,
    xs_lens=x_len,
    chunk_size=64,
    left_context_size=128,
    right_context_size=128,
)

print("feature: ", feature.shape)
print("feature_len: ", feature_len)
```
### Python API
#### Classification

ChunkFormer also supports speech classification tasks (e.g., gender, dialect, emotion, age recognition).

```python
from chunkformer import ChunkFormerModel

# Load a pre-trained classification model from Hugging Face or local directory
model = ChunkFormerModel.from_pretrained("path/to/classification/model")

# Single audio classification
result = model.classify_audio(
    audio_path="path/to/audio.wav",
    chunk_size=-1,  # -1 for full attention
    left_context_size=-1,
    right_context_size=-1,
)

print(result)
```

#### Transcription
```python
from chunkformer import ChunkFormerModel

# Load a pre-trained encoder from Hugging Face or local directory
model = ChunkFormerModel.from_pretrained("khanhld/chunkformer-ctc-large-vie")

# For single long-form audio transcription
transcription = model.endless_decode(
    audio_path="path/to/long_audio.wav",
    chunk_size=64,
    left_context_size=128,
    right_context_size=128,
    total_batch_duration=14400,  # in seconds
    return_timestamps=True
)
print(transcription)

# For batch processing of multiple audio files
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
transcriptions = model.batch_decode(
    audio_paths=audio_files,
    chunk_size=64,
    left_context_size=128,
    right_context_size=128,
    total_batch_duration=1800  # Total batch duration in seconds
)

for i, transcription in enumerate(transcriptions):
    print(f"Audio {i+1}: {transcription}")

```

### Command Line
#### Long-Form Audio Transcription
To test the model with a single [long-form audio file](samples/audios/audio_1.wav). Audio file extensions ".mp3", ".wav", ".flac", ".m4a", ".aac" are accepted:
```bash
chunkformer-decode \
    --model_checkpoint path/to/hf/checkpoint/repo \
    --audio_file path/to/audio.wav \
    --total_batch_duration 14400 \
    --chunk_size 64 \
    --left_context_size 128 \
    --right_context_size 128
```
Example Output:
```
[00:00:01.200] - [00:00:02.400]: this is a transcription example
[00:00:02.500] - [00:00:03.700]: testing the long-form audio
```

#### Batch Audio Transcription
The [data.tsv](samples/data.tsv) file must have at least one column named **wav**. Optionally, a column named **txt** can be included to compute the **Word Error Rate (WER)**. Output will be saved to the same file.

```bash
chunkformer-decode \
    --model_checkpoint path/to/hf/checkpoint/repo \
    --audio_list path/to/data.tsv \
    --total_batch_duration 14400 \
    --chunk_size 64 \
    --left_context_size 128 \
    --right_context_size 128
```
Example Output:
```
WER: 0.1234
```

#### Classification
To classify a single audio file:
```bash
chunkformer-decode \
    --model_checkpoint path/to/classification/model \
    --audio_file path/to/audio.wav
```

---

<a name = "training" ></a>
## Training

See **[🚀 Training Guide 🚀](examples/)** for complete documentation.


<a name = "citation" ></a>
## Citation
If you use this work in your research, please cite:

```bibtex
@INPROCEEDINGS{10888640,
  author={Le, Khanh and Ho, Tuan Vu and Tran, Dung and Chau, Duc Thanh},
  booktitle={ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  title={ChunkFormer: Masked Chunking Conformer For Long-Form Speech Transcription},
  year={2025},
  volume={},
  number={},
  pages={1-5},
  keywords={Scalability;Memory management;Graphics processing units;Signal processing;Performance gain;Hardware;Resource management;Speech processing;Standards;Context modeling;chunkformer;masked batch;long-form transcription},
  doi={10.1109/ICASSP49660.2025.10888640}}

```

<a name = "acknowledgments" ></a>
## Acknowledgments
This implementation is based on the WeNet framework. We extend our gratitude to the WeNet development team for providing an excellent foundation for speech recognition research and development.

---
