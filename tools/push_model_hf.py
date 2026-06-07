#!/usr/bin/env python3
"""
Python script to create and push ChunkFormer models to Hugging Face Hub.
This script handles the complete workflow of setting up a model repository
and pushing the trained ChunkFormer model with all necessary files.
"""

import argparse
import os
import sys
from typing import Optional

import yaml
from huggingface_hub import HfApi, create_repo, upload_folder


class ChunkFormerHubUploader:
    """Handler for uploading ChunkFormer models to Hugging Face Hub."""

    def __init__(self, token: Optional[str] = None):
        """
        Initialize the uploader.

        Args:
            token: Hugging Face token. If None, will try to use saved token.
        """
        self.api = HfApi(token=token)
        self.token = token

    def detect_model_type(self, model_dir: str) -> tuple[str, dict]:
        """
        Detect whether the model is ASR or Classification based on config.

        Args:
            model_dir: Directory containing the model files

        Returns:
            Tuple of (model_type, tasks_info)
            - model_type: "asr" or "classification"
            - tasks_info: Dictionary with task information (for classification)
        """
        config_path = os.path.join(model_dir, "config.yaml")

        if not os.path.exists(config_path):
            print(f"Warning: config.yaml not found in {model_dir}, assuming ASR model")
            return "asr", {}

        try:
            with open(config_path, "r") as f:
                config = yaml.load(f, Loader=yaml.FullLoader)

            # Check the model type
            model_type_str = config.get("model", "asr_model").lower()

            if model_type_str in ("vipvl", "bestrq"):
                # Self-supervised pretrained encoder (ViP-VL). "bestrq" is a
                # legacy alias for configs produced before the rename.
                return "vipvl", {}

            if "classification" in model_type_str:
                # Extract task information
                tasks_info = {}
                if "model_conf" in config:
                    tasks_conf = config["model_conf"].get("tasks", {})
                    for task_name, num_classes in tasks_conf.items():
                        tasks_info[task_name] = num_classes

                return "classification", tasks_info
            else:
                return "asr", {}

        except Exception as e:
            print(f"Warning: Error reading config.yaml: {e}, assuming ASR model")
            return "asr", {}

    def create_asr_model_card(self, repo_id: str) -> str:
        """Create model card for ASR model."""
        model_card = f"""---
tags:
- speech-recognition
- audio
- chunkformer
- ctc
- pytorch
- transformers
- automatic-speech-recognition
- long-form transcription
- asr
license: apache-2.0
library_name: transformers
pipeline_tag: automatic-speech-recognition
---

# ChunkFormer ASR Model
<style>
img {{
display: inline;
}}
</style>
[![GitHub](https://img.shields.io/badge/GitHub-ChunkFormer-blue)](https://github.com/khanld/chunkformer)
[![Paper](https://img.shields.io/badge/Paper-ICASSP%202025-green)](https://arxiv.org/abs/2502.14673)


## Usage

Install the package:

```bash
pip install chunkformer
```

### Long-Form Audio Transcription

```python
from chunkformer import ChunkFormerModel

# Load the model
model = ChunkFormerModel.from_pretrained("{repo_id}")

# For long-form audio transcription with timestamps
transcription = model.endless_decode(
    audio_path="path/to/your/audio.wav",
    chunk_size=64,
    left_context_size=128,
    right_context_size=128,
    return_timestamps=True
)
print(transcription)
```

### Batch Processing

```python
# For batch processing multiple audio files
audio_files = ["audio1.wav", "audio2.wav", "audio3.wav"]
transcriptions = model.batch_decode(
    audio_paths=audio_files,
    chunk_size=64,
    left_context_size=128,
    right_context_size=128
)

for i, transcription in enumerate(transcriptions):
    print(f"Audio {{i+1}}: {{transcription}}")
```

## Training

This model was trained using the ChunkFormer framework. For more details about the training process and to access the source code, please visit: https://github.com/khanld/chunkformer

Paper: https://arxiv.org/abs/2502.14673

## Citation

If you use this work in your research, please cite:

```bibtex
@INPROCEEDINGS{{10888640,
    author={{Le, Khanh and Ho, Tuan Vu and Tran, Dung and Chau, Duc Thanh}},
    booktitle={{ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}},
    title={{ChunkFormer: Masked Chunking Conformer For Long-Form Speech Transcription}},
    year={{2025}},
    volume={{}},
    number={{}},
    pages={{1-5}},
    keywords={{Scalability;Memory management;Graphics processing units;Signal processing;Performance gain;Hardware;Resource management;Speech processing;Standards;Context modeling;chunkformer;masked batch;long-form transcription}},
    doi={{10.1109/ICASSP49660.2025.10888640}}}}
```
"""  # noqa: E501
        return model_card

    def create_classification_model_card(self, repo_id: str, tasks_info: dict) -> str:
        """Create model card for Classification model."""
        # Build tasks description
        tasks_desc = ""
        if tasks_info:
            tasks_desc = "\n## Classification Tasks\n\n"
            for task_name, num_classes in tasks_info.items():
                tasks_desc += f"- **{task_name.capitalize()}**: {num_classes} classes\n"

        # Build task tags
        task_tags = ""
        if tasks_info:
            for task_name in tasks_info.keys():
                task_tags += f"- {task_name.lower()}\n"
        model_card = f"""---
tags:
- audio-classification
- speech-classification
- audio
- chunkformer
- pytorch
- transformers
- speech-processing
{task_tags}
license: apache-2.0
library_name: transformers
pipeline_tag: audio-classification
---

# ChunkFormer Classification Model
<style>
img {{
display: inline;
}}
</style>
[![GitHub](https://img.shields.io/badge/GitHub-ChunkFormer-blue)](https://github.com/khanld/chunkformer)
[![Paper](https://img.shields.io/badge/Paper-ICASSP%202025-green)](https://arxiv.org/abs/2502.14673)

This model performs speech classification tasks such as gender recognition, dialect identification, emotion detection, and age classification.
{tasks_desc}

## Usage

Install the package:

```bash
pip install chunkformer
```

### Single Audio Classification

```python
from chunkformer import ChunkFormerModel

# Load the model
model = ChunkFormerModel.from_pretrained("{repo_id}")

# Classify a single audio file
result = model.classify_audio(
    audio_path="path/to/your/audio.wav",
    chunk_size=-1,  # -1 for full attention
    left_context_size=-1,
    right_context_size=-1
)

print(result)
# Output example:
# {{
#   'gender': {{
#       'label': 'female',
#       'label_id': 0,
#       'prob': 0.95
#   }},
#   'dialect': {{
#       'label': 'northern dialect',
#       'label_id': 3,
#       'prob': 0.70
#   }},
#   'emotion': {{
#       'label': 'neutral',
#       'label_id': 5,
#       'prob': 0.80
#   }}
# }}
```

### Command Line Usage

```bash
chunkformer-decode \\
    --model_checkpoint {repo_id} \\
    --audio_file path/to/audio.wav
```

## Training

This model was trained using the ChunkFormer framework. For more details about the training process and to access the source code, please visit: https://github.com/khanld/chunkformer

Paper: https://arxiv.org/abs/2502.14673

## Citation

If you use this work in your research, please cite:

```bibtex
@INPROCEEDINGS{{10888640,
    author={{Le, Khanh and Ho, Tuan Vu and Tran, Dung and Chau, Duc Thanh}},
    booktitle={{ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}},
    title={{ChunkFormer: Masked Chunking Conformer For Long-Form Speech Transcription}},
    year={{2025}},
    volume={{}},
    number={{}},
    pages={{1-5}},
    keywords={{Scalability;Memory management;Graphics processing units;Signal processing;Performance gain;Hardware;Resource management;Speech processing;Standards;Context modeling;chunkformer;masked batch;long-form transcription}},
    doi={{10.1109/ICASSP49660.2025.10888640}}}}
```
"""  # noqa: E501
        return model_card

    def create_ssl_model_card(self, repo_id: str) -> str:
        """Create model card for the ViP-VL self-supervised pretrained encoder."""
        model_card = f"""---
tags:
- speech
- self-supervised-learning
- vip-vl
- best-rq
- chunkformer
- pretrained-encoder
- speech-pretraining
- pytorch
language:
- vi
license: apache-2.0
library_name: transformers
---

# ViP-VL: **Vi**etnamese Self-supervised speech **P**retraining model leveraging **V**ector-quantization **L**earning
<style>
img {{
display: inline;
}}
</style>
[![GitHub](https://img.shields.io/badge/GitHub-ChunkFormer-blue)](https://github.com/khanld/chunkformer)
[![Paper](https://img.shields.io/badge/Paper-INTERSPEECH%202026-green)](https://github.com/khanld/chunkformer)

**ViP-VL** is a self-supervised speech pretraining model for Vietnamese, accepted to
**INTERSPEECH 2026**. This repository hosts the pretrained **ViP-VL** model: a ChunkFormer
encoder pretrained on large-scale unlabeled Vietnamese speech with a random-projection-quantizer
masked-prediction objective ([BEST-RQ](https://arxiv.org/abs/2202.01855)). It is designed to
initialize downstream finetuning (ASR / RNN-T / classification).

## Method

ViP-VL adapts the random-projection-quantizer masked-prediction recipe (BEST-RQ) to an
aggressive **8× temporal-subsampling** ChunkFormer backbone, fixing the synchronization
between the masking manifold and the encoder's subsampling rate:

- **Masking** is applied to the raw 10 ms log-mel frames *before* subsampling; a subsampled
  frame is treated as masked iff **≥ 80 %** of its constituent input frames are masked.
- **Targets** come from a *frozen* random-projection quantizer: a fixed random projection of
  the (CMVN-normalized) input is matched by L2 nearest-neighbour to a fixed random codebook
  (1024 entries, dimension 16); the encoder is trained with a masked language-model (NLL)
  objective over masked positions.

## Architecture

| | |
|---|---|
| Encoder | ChunkFormer |
| Encoder blocks | 12 |
| Hidden size | 512 |
| Attention heads | 8 |
| FFN size | 2048 |
| CNN module kernel | 15 |
| Subsampling | `dw_striding` (8×) |
| Positional encoding | chunk relative |
| Input features | 80-dim log-mel fbank @ 16 kHz |

## Files

- `pytorch_model.pt` — encoder-only state dict (`encoder.*`).
- `config.yaml` — encoder configuration (`encoder_conf`) and feature settings.
- `global_cmvn` — global CMVN statistics used during pretraining.

## Finetuning

The encoder weights load with `strict=False`, so point any ChunkFormer ASR / RNN-T /
classification recipe at this checkpoint and train the task heads from scratch. Make sure
the downstream `encoder_conf` matches `config.yaml`.

The `checkpoint` argument accepts **either a local path or this repo id directly** —
`load_checkpoint` looks for a local file/directory first and otherwise downloads
`pytorch_model.pt` from the Hub automatically (cached locally), so no manual download
step is required:

```bash
# e.g. in examples/asr/ctc/run.sh (or rnnt / classification)

# Option A — download straight from the Hub (recommended)
checkpoint={repo_id}

# Option B — local path to an exported bundle
checkpoint=/path/to/{repo_id}/pytorch_model.pt
```

For a **private** repo, authenticate first with `huggingface-cli login` or by exporting
`HF_TOKEN`. To pre-download (or inspect) the files manually:

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(repo_id="{repo_id}")
# local_dir/pytorch_model.pt  ->  also valid as the finetuning `checkpoint=`
```

## Citation

If you use this model, please cite ViP-VL (INTERSPEECH 2026) and ChunkFormer:

```bibtex
@inproceedings{{vipvl,
    title={{ViP-VL: Vietnamese Self-supervised Speech Pretraining Model Leveraging Vector-Quantization Learning}},
    author={{Le, Khanh and Hoang, Kiet Anh and Nguyen, Bao and Vo, Duy and Vo, Dung and Tran, Thai and Pham, Linh and Doan, Khoa D}},
    booktitle={{Proc. INTERSPEECH 2026}},
    year={{2026}}
}}

@INPROCEEDINGS{{10888640,
    author={{Le, Khanh and Ho, Tuan Vu and Tran, Dung and Chau, Duc Thanh}},
    booktitle={{ICASSP 2025 - 2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}},
    title={{ChunkFormer: Masked Chunking Conformer For Long-Form Speech Transcription}},
    year={{2025}},
    pages={{1-5}},
    doi={{10.1109/ICASSP49660.2025.10888640}}}}
```
"""  # noqa: E501
        return model_card

    def create_model_card(self, model_dir: str, repo_id: str) -> str:
        """
        Create a model card for the ChunkFormer model (ASR / Classification / ViP-VL).

        Args:
            model_dir: Directory containing the model files
            repo_id: Repository ID on Hugging Face

        Returns:
            Model card content as string
        """
        # Detect model type
        model_type, tasks_info = self.detect_model_type(model_dir)

        print(f"Detected model type: {model_type}")
        if tasks_info:
            print(f"Classification tasks: {tasks_info}")

        # Generate appropriate model card
        if model_type == "vipvl":
            return self.create_ssl_model_card(repo_id)
        elif model_type == "classification":
            return self.create_classification_model_card(repo_id, tasks_info)
        else:
            return self.create_asr_model_card(repo_id)

    def create_repository(self, repo_id: str, private: bool = False) -> bool:
        """
        Create a new repository on Hugging Face Hub.

        Args:
            repo_id: Repository ID (username/repo-name)
            private: Whether to create a private repository

        Returns:
            True if repository was created or already exists, False otherwise
        """
        try:
            # create_repo with exist_ok=True is idempotent: it creates the repo
            # if missing and is a no-op if it already exists. This avoids a
            # separate repo_info() existence probe, which reports private repos
            # as "not found" when the token is not applied and is therefore
            # unreliable for gating creation.
            print(f"Creating repository (if needed): {repo_id}")
            create_repo(
                repo_id=repo_id,
                token=self.token,
                private=private,
                repo_type="model",
                exist_ok=True,
            )
            print(f"✓ Repository ready: {repo_id}")
            return True

        except Exception as e:
            print(f"✗ Failed to create repository {repo_id}: {e}")
            return False

    def upload_model(
        self, model_dir: str, repo_id: str, commit_message: Optional[str] = None
    ) -> bool:
        """
        Upload model files to Hugging Face Hub (directly from model_dir).

        Args:
            model_dir: Directory containing model files
            repo_id: Repository ID on Hugging Face Hub
            commit_message: Commit message for the upload

        Returns:
            True if upload successful, False otherwise
        """
        try:
            # Create model card
            model_card_content = self.create_model_card(model_dir, repo_id)
            model_card_path = os.path.join(model_dir, "README.md")
            with open(model_card_path, "w", encoding="utf-8") as f:
                f.write(model_card_content)
            print(f"✓ Created model card: {model_card_path}")

            # Upload all files
            commit_msg = commit_message or f"Upload ChunkFormer model from {model_dir}"
            print(f"Uploading model to {repo_id}...")

            upload_folder(
                folder_path=model_dir,
                repo_id=repo_id,
                token=self.token,
                commit_message=commit_msg,
                repo_type="model",
            )

            print(f"✓ Model uploaded successfully to: https://huggingface.co/{repo_id}")
            return True

        except Exception as e:
            print(f"✗ Failed to upload model: {e}")
            return False

    def push_model_from_checkpoint_dir(
        self,
        checkpoint_dir: str,
        repo_id: str,
        private: bool = False,
        commit_message: Optional[str] = None,
    ) -> bool:
        """
        Complete workflow to push a model from checkpoint directory to Hugging Face Hub.

        Args:
            checkpoint_dir: Directory containing model checkpoint and files
            repo_id: Repository ID (username/repo-name)
            private: Whether to create a private repository
            commit_message: Commit message for the upload

        Returns:
            True if successful, False otherwise
        """
        print(f"Starting upload process for {checkpoint_dir} -> {repo_id}")

        # Validate checkpoint directory
        if not os.path.exists(checkpoint_dir):
            print(f"✗ Checkpoint directory does not exist: {checkpoint_dir}")
            return False

        model_file = os.path.join(checkpoint_dir, "pytorch_model.pt")
        if not os.path.exists(model_file):
            print(f"✗ Model checkpoint not found: {model_file}")
            return False

        # Create repository
        if not self.create_repository(repo_id, private):
            return False

        # Upload model
        if not self.upload_model(checkpoint_dir, repo_id, commit_message):
            return False

        print("🎉 Successfully pushed model to Hugging Face Hub!")
        print(f"Model URL: https://huggingface.co/{repo_id}")
        print("\nYou can now use your model with:")
        print("from chunkformer import ChunkFormerModel")
        print(f"model = ChunkFormerModel.from_pretrained('{repo_id}')")

        return True


def main():
    """Main function to handle command line arguments and run the upload process."""
    parser = argparse.ArgumentParser(
        description="Upload ChunkFormer model to Hugging Face Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing the model checkpoint and files (stage 5)",
    )

    parser.add_argument(
        "--repo_id", type=str, required=True, help="Hugging Face repository ID (username/repo-name)"
    )

    parser.add_argument(
        "--token", type=str, default=None, help="Hugging Face token (optional if already logged in)"
    )

    parser.add_argument("--private", action="store_true", help="Create a private repository")

    parser.add_argument(
        "--commit_message", type=str, default=None, help="Custom commit message for the upload"
    )

    args = parser.parse_args()

    # Validate arguments
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory does not exist: {args.model_dir}")
        sys.exit(1)

    if "/" not in args.repo_id:
        print(f"Error: Repository ID must be in format 'username/repo-name', got: {args.repo_id}")
        sys.exit(1)

    # Initialize uploader
    try:
        uploader = ChunkFormerHubUploader(token=args.token)
    except Exception as e:
        print(f"Error: Failed to initialize Hugging Face API: {e}")
        print("Make sure you have a valid Hugging Face token.")
        print("You can login with: huggingface-cli login")
        sys.exit(1)

    # Push model
    success = uploader.push_model_from_checkpoint_dir(
        checkpoint_dir=args.model_dir,
        repo_id=args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
    )

    if not success:
        print("Upload failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
