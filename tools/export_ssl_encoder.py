#!/usr/bin/env python3
"""Export a finetune-ready, encoder-only checkpoint from a ViP-VL SSL run.

A ViP-VL checkpoint contains the encoder plus the self-supervised heads
(``quantizer.*``, ``final_proj.*``, ``mask_emb``). For downstream finetuning we
only need the encoder weights: ChunkFormer's ``load_checkpoint`` uses
``strict=False``, so an ``encoder.*``-only checkpoint loads cleanly into an ASR /
RNN-T / classification model built with the same ``encoder_conf`` while the new
task heads are initialised from scratch.

This script:
  1. keeps only the ``encoder.*`` tensors and writes ``pytorch_model.pt``;
  2. writes a sanitised ``config.yaml`` (no training data paths / run dirs);
  3. copies the global CMVN file;
  4. emits a finetuning-focused ``README.md``.

The resulting directory can be loaded with ``ChunkFormerModel.from_pretrained``
or uploaded to the Hugging Face Hub with ``tools/push_model_hf.py``.

Usage:
    python tools/export_ssl_encoder.py \
        --checkpoint exp/vipvl/avg_50.pt \
        --config exp/vipvl/train.yaml \
        --cmvn data/train/global_cmvn \
        --output_dir exp/vipvl/encoder_checkpoint
"""

import argparse
import os
import shutil

import torch
import yaml

ENCODER_PREFIX = "encoder."

# Fields that reference the private training environment / data and must not be
# leaked in a published checkpoint config.
SENSITIVE_KEYS = {
    "model_dir",
    "tensorboard_dir",
    "init_infos",
    "save_time",
    "tag",
    "save_states",
    "train_data",
    "cv_data",
    "data_type",
}


def extract_encoder_state_dict(checkpoint_path: str) -> dict:
    state = torch.load(checkpoint_path, map_location="cpu")
    # Some checkpoints nest the weights under a key.
    if "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]

    encoder_state = {k: v for k, v in state.items() if k.startswith(ENCODER_PREFIX)}
    dropped = sorted({k.split(".")[0] for k in state if not k.startswith(ENCODER_PREFIX)})

    if not encoder_state:
        raise ValueError(
            f"No '{ENCODER_PREFIX}*' tensors found in {checkpoint_path}. "
            "Is this a ChunkFormer ViP-VL checkpoint?"
        )

    print(f"Kept {len(encoder_state)} encoder tensors; dropped SSL-head groups: {dropped}")
    return encoder_state


def sanitize_config(config: dict) -> dict:
    """Strip training-environment / data-specific fields and genericise the
    CMVN path so the published config does not leak private data info."""
    clean = {k: v for k, v in config.items() if k not in SENSITIVE_KEYS}

    # Point CMVN to the file shipped alongside the checkpoint (filename only).
    if clean.get("cmvn") == "global_cmvn":
        cmvn_conf = dict(clean.get("cmvn_conf", {}))
        cmvn_conf["cmvn_file"] = "global_cmvn"
        cmvn_conf.setdefault("is_json_cmvn", True)
        clean["cmvn_conf"] = cmvn_conf

    return clean


def write_readme(output_dir: str, repo_hint: str = "<user/repo>"):
    readme = f"""---
tags:
- speech
- self-supervised-learning
- vip-vl
- best-rq
- chunkformer
- pretrained-encoder
- pytorch
license: apache-2.0
library_name: transformers
---

# ChunkFormer ViP-VL Pretrained Encoder

Self-supervised (ViP-VL) pretrained ChunkFormer **encoder** weights, intended as
an initialisation for downstream finetuning (ASR / RNN-T / classification).

This checkpoint contains **only** the `encoder.*` weights. The self-supervised
heads (random-projection quantizer, prediction head, mask embedding) are not
included because they are not needed for finetuning.

## Files
- `pytorch_model.pt` — encoder-only state dict (`encoder.*`).
- `config.yaml` — encoder configuration (`encoder_conf`) and feature settings.
- `global_cmvn` — global CMVN statistics used during pretraining.

## Finetuning

Point a ChunkFormer ASR / RNN-T / classification recipe at this checkpoint. The
encoder weights load via `strict=False`; the task-specific decoder/CTC/heads are
trained from scratch. Make sure your finetuning `encoder_conf` matches `config.yaml`.

The `checkpoint` argument accepts **either a local path or this repo id directly** —
`load_checkpoint` resolves a local file/directory first and otherwise downloads
`pytorch_model.pt` from the Hub automatically (cached locally):

```bash
# In your downstream recipe's run.sh

# Download straight from the Hub (recommended)
checkpoint={repo_hint}

# Or a local path to this bundle
checkpoint=/path/to/pytorch_model.pt
```

Or load programmatically:

```python
from huggingface_hub import hf_hub_download
import torch

ckpt = hf_hub_download("{repo_hint}", "pytorch_model.pt")
state = torch.load(ckpt, map_location="cpu")
missing, unexpected = model.encoder.load_state_dict(
    {{k[len("encoder."):]: v for k, v in state.items()}}, strict=False
)
```

## Citation

```bibtex
@INPROCEEDINGS{{10888640,
    author={{Le, Khanh and Ho, Tuan Vu and Tran, Dung and Chau, Duc Thanh}},
    booktitle={{ICASSP 2025 - 2025 IEEE International Conference on Acoustics,
        Speech and Signal Processing (ICASSP)}},
    title={{ChunkFormer: Masked Chunking Conformer For Long-Form Speech Transcription}},
    year={{2025}},
    pages={{1-5}},
    doi={{10.1109/ICASSP49660.2025.10888640}}}}
```
"""
    with open(os.path.join(output_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--checkpoint", required=True, help="ViP-VL checkpoint (.pt)")
    parser.add_argument(
        "--config", required=True, help="Training config yaml (e.g. exp/.../train.yaml)"
    )
    parser.add_argument("--cmvn", default=None, help="global_cmvn file to bundle")
    parser.add_argument("--output_dir", required=True, help="Output bundle directory")
    parser.add_argument(
        "--repo_hint", default="<user/repo>", help="Repo id used in the README example"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. encoder-only weights
    encoder_state = extract_encoder_state_dict(args.checkpoint)
    out_ckpt = os.path.join(args.output_dir, "pytorch_model.pt")
    torch.save(encoder_state, out_ckpt)
    print(f"Wrote encoder checkpoint: {out_ckpt}")

    # 2. sanitised config
    with open(args.config, "r") as fin:
        config = yaml.safe_load(fin)
    clean_config = sanitize_config(config)
    out_cfg = os.path.join(args.output_dir, "config.yaml")
    with open(out_cfg, "w") as fout:
        yaml.dump(clean_config, fout, sort_keys=True)
    print(f"Wrote sanitised config: {out_cfg}")

    # 3. CMVN
    if args.cmvn and os.path.exists(args.cmvn):
        shutil.copy(args.cmvn, os.path.join(args.output_dir, "global_cmvn"))
        print(f"Copied CMVN: {args.cmvn} -> {args.output_dir}/global_cmvn")
    else:
        print("Warning: CMVN file not provided or not found; skipping.")

    # 4. README
    write_readme(args.output_dir, args.repo_hint)
    print("Wrote README.md")

    print("\nEncoder bundle ready:")
    for name in sorted(os.listdir(args.output_dir)):
        print(f"  {name}")


if __name__ == "__main__":
    main()
