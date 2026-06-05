#!/usr/bin/env python3
"""Deterministic parity check for the BestRQ SSL model.

Builds a BestRQ model from a training config, loads a checkpoint, runs a single
forward pass on a fixed synthetic batch and prints the resulting metrics. The
random-projection masking is made deterministic by seeding numpy/torch and by
forcing ``numpy.random.default_rng`` to a fixed seed, so the same code+checkpoint
always yields identical numbers. This lets us confirm that the cleaned BestRQ
branch reproduces the original ``kl/add_wav2vec2_ssl`` results exactly.

Usage:
    PYTHONPATH=<repo> python tools/verify_bestrq_parity.py \
        --config <path/to/init.yaml> --checkpoint <path/to/avg_50.pt>
"""

import argparse

import numpy as np
import torch
import yaml

from chunkformer.utils.checkpoint import load_checkpoint
from chunkformer.utils.init_model import init_speech_model


def set_determinism(seed: int = 0, rng_seed: int = 12345):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    # compute_mask_indices uses np.random.default_rng() with no explicit seed,
    # which would otherwise draw fresh entropy on every call. Force a fixed
    # generator so masking is reproducible across runs and branches.
    _orig = np.random.default_rng

    def _seeded(*args, **kwargs):
        return _orig(rng_seed)

    np.random.default_rng = _seeded


def build_model(config_path: str):
    with open(config_path, "r") as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    # Skip CMVN so we do not depend on an external global_cmvn file. The CMVN
    # buffers from the checkpoint are ignored identically on both branches.
    configs["cmvn"] = None
    configs.pop("cmvn_conf", None)
    configs.setdefault("input_dim", 80)
    model, _ = init_speech_model(args=None, configs=configs)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Training config yaml (e.g. init.yaml)")
    parser.add_argument("--checkpoint", required=True, help="BestRQ checkpoint (.pt)")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--frames", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_determinism(seed=args.seed)

    model = build_model(args.config)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    feat_dim = 80
    feats = torch.randn(args.batch, args.frames, feat_dim, generator=None)
    feats_lengths = torch.full((args.batch,), args.frames, dtype=torch.int32)
    batch = {"feats": feats, "feats_lengths": feats_lengths}

    with torch.no_grad():
        out = model(batch, torch.device("cpu"))

    def scalar(v):
        if torch.is_tensor(v):
            return float(v.detach().float().cpu().item())
        return float(v)

    print("=== BestRQ parity metrics ===")
    for key in ["loss", "corr", "ntokens", "mask_percentile", "code_perplexity", "prob_perplexity"]:
        if key in out:
            print(f"{key}: {scalar(out[key]):.10f}")


if __name__ == "__main__":
    main()
