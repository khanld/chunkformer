#!/usr/bin/env python3
# Copyright (c) 2024 ChunkFormer Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Inference script for speech classification tasks."""

import argparse
import json
import logging
import os
from contextlib import nullcontext

import torch
import yaml
from tqdm import tqdm

from chunkformer.dataset.dataset import Dataset
from chunkformer.utils.checkpoint import load_checkpoint
from chunkformer.utils.init_model import init_speech_model


def get_args():
    parser = argparse.ArgumentParser(description="Classification inference")
    parser.add_argument("--gpu", type=int, default=0, help="GPU id, -1 for CPU")
    parser.add_argument("--config", required=True, help="Config file")
    parser.add_argument("--data_type", default="raw", choices=["raw", "shard"], help="Data type")
    parser.add_argument("--test_data", required=True, help="Test data list file")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--result_dir", required=True, help="Result directory")
    parser.add_argument("--chunk_size", type=int, default=-1, help="Chunk size for encoder")
    parser.add_argument("--left_context_size", type=int, default=-1, help="Left context size")
    parser.add_argument("--right_context_size", type=int, default=-1, help="Right context size")
    parser.add_argument(
        "--dtype", default="fp32", choices=["fp16", "fp32"], help="Data type for inference"
    )
    return parser.parse_args()


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # Setup device
    if args.gpu < 0:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.gpu}")

    # Load config
    with open(args.config, "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    # Get tasks
    tasks = list(configs.get("model_conf", {}).get("tasks", {}).keys())
    if not tasks:
        logging.error("No tasks defined in config")
        return

    logging.info(f"Tasks: {tasks}")

    # Initialize model
    model, _ = init_speech_model(args, configs)
    load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    logging.info(f"Model loaded from {args.checkpoint}")

    # Setup dataset
    dataset_conf = configs.get("dataset_conf", {})
    dataset_conf["shuffle"] = False
    dataset_conf["sort"] = False
    dataset_conf["batch_size"] = args.batch_size
    dataset_conf["batch_type"] = "static"

    test_dataset = Dataset(
        args.data_type,
        args.test_data,
        tokenizer=None,
        conf=dataset_conf,
        partition=False,
    )

    # Dataset already handles batching internally via padding function
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=None,
        num_workers=0,
    )

    # Create output directory
    os.makedirs(args.result_dir, exist_ok=True)

    # Output files
    pred_file = os.path.join(args.result_dir, "predictions.tsv")
    detail_file = os.path.join(args.result_dir, "predictions_detail.jsonl")

    # Setup dtype
    if args.dtype == "fp16":
        dtype = torch.float16
        autocast_context = torch.cuda.amp.autocast
    else:
        dtype = torch.float32
        autocast_context = nullcontext

    # Run inference
    logging.info("Starting inference...")
    all_predictions = []

    with torch.no_grad(), autocast_context():
        for batch_idx, batch in enumerate(tqdm(test_data_loader)):
            # Get keys from batch
            keys = batch.get(
                "keys", [f"utt_{batch_idx}_{i}" for i in range(batch["feats"].size(0))]
            )

            # Move to device
            feats = batch["feats"].to(device, dtype=dtype)
            feats_lengths = batch["feats_lengths"].to(device)

            # Forward pass
            results = model.classify(
                feats,
                feats_lengths,
                chunk_size=args.chunk_size,
                left_context_size=args.left_context_size,
                right_context_size=args.right_context_size,
            )

            # Process results
            batch_size = feats.size(0)
            for i in range(batch_size):
                key = keys[i] if i < len(keys) else f"utt_{batch_idx}_{i}"

                pred_dict = {"key": key}

                for task in tasks:
                    pred_key = f"{task}_prediction"
                    prob_key = f"{task}_probability"

                    prediction = results[pred_key][i].item()
                    pred_dict[task] = prediction

                    probability = results[prob_key][i].item()
                    pred_dict[f"{task}_prob"] = probability

                all_predictions.append(pred_dict)

    # Save predictions in TSV format
    logging.info(f"Saving predictions to {pred_file}")
    with open(pred_file, "w", encoding="utf-8") as f:
        # Write header
        header = ["key"] + tasks
        f.write("\t".join(header) + "\n")

        # Write predictions
        for pred in all_predictions:
            row = [pred["key"]] + [str(pred.get(task, "-1")) for task in tasks]
            f.write("\t".join(row) + "\n")

    # Save detailed predictions in JSONL format
    logging.info(f"Saving detailed predictions to {detail_file}")
    with open(detail_file, "w", encoding="utf-8") as f:
        for pred in all_predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    logging.info("Inference complete!")
    logging.info(f"Total samples: {len(all_predictions)}")
    logging.info(f"Results saved to: {args.result_dir}")


if __name__ == "__main__":
    main()
