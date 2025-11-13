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
from huggingface_hub.utils import RepositoryNotFoundError


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

            # Check if it's a classification model
            model_type_str = config.get("model", "asr_model")

            if "classification" in model_type_str.lower():
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

    def create_model_card(self, model_dir: str, repo_id: str) -> str:
        """
        Create a model card for the ChunkFormer model (ASR or Classification).

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
        if model_type == "classification":
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
            # Check if repository already exists
            try:
                self.api.repo_info(repo_id)
                print(f"✓ Repository already exists: {repo_id}")
                return True
            except RepositoryNotFoundError:
                pass

            # Create new repository
            print(f"Creating new repository: {repo_id}")
            create_repo(repo_id=repo_id, token=self.token, private=private, repo_type="model")
            print(f"✓ Repository created successfully: {repo_id}")
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
