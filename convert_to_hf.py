#!/usr/bin/env python3
"""
Convert ChunkFormer models to Hugging Face format
"""
import os
import argparse
import shutil
import yaml
import json
from pathlib import Path

from chunkformer_hf import ChunkFormerConfig, ChunkFormerForASR


def convert_chunkformer_to_hf(
    original_model_path: str,
    output_path: str,
    model_name: str = "chunkformer"
):
    """
    Convert original ChunkFormer model to Hugging Face format.
    
    Args:
        original_model_path: Path to original ChunkFormer model directory
        output_path: Path where to save the HF-compatible model
        model_name: Name for the model
    """
    print(f"Converting ChunkFormer model from {original_model_path} to {output_path}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Load original config
    config_yaml_path = os.path.join(original_model_path, "config.yaml")
    if not os.path.exists(config_yaml_path):
        raise FileNotFoundError(f"config.yaml not found in {original_model_path}")
    
    # Create HF config from YAML
    config = ChunkFormerConfig.from_yaml_config(config_yaml_path)
    
    # Save HF config
    config.save_pretrained(output_path)
    
    # Copy necessary files
    files_to_copy = [
        "pytorch_model.bin",
        "avg_75.pt",  # Alternative checkpoint name
        "vocab.txt",
        "global_cmvn",  # CMVN file if exists
    ]
    
    for file_name in files_to_copy:
        src_path = os.path.join(original_model_path, file_name)
        if os.path.exists(src_path):
            dst_path = os.path.join(output_path, file_name)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, dst_path)
                print(f"Copied {file_name}")
            elif os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                print(f"Copied directory {file_name}")
    
    # If pytorch_model.bin doesn't exist but avg_75.pt does, rename it
    pytorch_model_path = os.path.join(output_path, "pytorch_model.bin")
    avg_model_path = os.path.join(output_path, "avg_75.pt")
    
    if not os.path.exists(pytorch_model_path) and os.path.exists(avg_model_path):
        shutil.copy2(avg_model_path, pytorch_model_path)
        print("Renamed avg_75.pt to pytorch_model.bin")
    
    # Create a simple README
    readme_content = f"""---
license: apache-2.0
language: 
- en
pipeline_tag: automatic-speech-recognition
tags:
- speech
- audio
- chunkformer
- asr
---

# {model_name}

This is a ChunkFormer model for Automatic Speech Recognition, converted to Hugging Face format.

## Usage

```python
from chunkformer_hf import ChunkFormerForASR
import torch

# Load the model
model = ChunkFormerForASR.from_pretrained("{output_path}")

# Prepare your audio features (log mel-spectrogram)
# features should be of shape (batch_size, seq_len, 80)
features = torch.randn(1, 1000, 80)  # Example

# Forward pass
outputs = model(features)
logits = outputs['logits']  # CTC logits
```

## Model Details

- **Model Type**: ChunkFormer for ASR
- **Language**: English/Vietnamese (depending on training data)
- **Input**: Log mel-spectrogram features (80-dimensional)
- **Output**: CTC logits for character/subword prediction

## Training

This model was trained using the original ChunkFormer training pipeline.
"""
    
    with open(os.path.join(output_path, "README.md"), "w") as f:
        f.write(readme_content)
    
    print(f"✅ Conversion completed! Model saved to {output_path}")
    print(f"You can now load it using: ChunkFormerForASR.from_pretrained('{output_path}')")


def test_converted_model(model_path: str):
    """Test the converted model to make sure it works."""
    print(f"Testing converted model at {model_path}")
    
    try:
        # Load the model
        model = ChunkFormerForASR.from_pretrained(model_path)
        model.eval()
        
        # Create dummy input
        batch_size, seq_len, feature_dim = 1, 100, 80
        features = torch.randn(batch_size, seq_len, feature_dim)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(features)
        
        print(f"✅ Model test passed!")
        print(f"   Input shape: {features.shape}")
        print(f"   Output logits shape: {outputs['logits'].shape}")
        print(f"   Output encoder shape: {outputs['encoder_outputs'].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert ChunkFormer model to Hugging Face format")
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to original ChunkFormer model directory"
    )
    parser.add_argument(
        "output_path", 
        type=str,
        help="Path where to save the HF-compatible model"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="chunkformer",
        help="Name for the model (used in README)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test the converted model after conversion"
    )
    
    args = parser.parse_args()
    
    # Convert the model
    convert_chunkformer_to_hf(
        args.input_path,
        args.output_path,
        args.model_name
    )
    
    # Test if requested
    if args.test:
        test_converted_model(args.output_path)


if __name__ == "__main__":
    main()
