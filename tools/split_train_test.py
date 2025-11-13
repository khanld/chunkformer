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

"""Split TSV file into train and test sets randomly."""

import argparse
import random
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Split TSV file into train and test sets randomly")
    parser.add_argument("-i", "--input", required=True, type=str, help="Input TSV file path")
    parser.add_argument(
        "-o",
        "--output-dir",
        required=True,
        type=str,
        help="Output directory for train and test files",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Ratio of test set (default: 0.2 for 20%%)",
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.0,
        help="Ratio of dev set (default: 0.0, no dev set)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle the data before splitting")

    args = parser.parse_args()

    # Validate ratios
    if args.test_ratio < 0 or args.test_ratio >= 1:
        print("Error: test_ratio must be between 0 and 1")
        sys.exit(1)
    if args.dev_ratio < 0 or args.dev_ratio >= 1:
        print("Error: dev_ratio must be between 0 and 1")
        sys.exit(1)
    if args.test_ratio + args.dev_ratio >= 1:
        print("Error: test_ratio + dev_ratio must be less than 1")
        sys.exit(1)

    # Set random seed
    random.seed(args.seed)

    # Read input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {args.input} does not exist")
        sys.exit(1)

    print(f"Reading data from: {args.input}")
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 2:
        print("Error: Input file must have at least a header and one data line")
        sys.exit(1)

    # Separate header and data
    header = lines[0]
    data_lines = lines[1:]

    print(f"Total samples: {len(data_lines)}")

    # Shuffle if requested
    if args.shuffle:
        print("Shuffling data...")
        random.shuffle(data_lines)

    # Calculate split sizes
    total_samples = len(data_lines)
    test_size = int(total_samples * args.test_ratio)
    dev_size = int(total_samples * args.dev_ratio)
    train_size = total_samples - test_size - dev_size

    print(f"Train samples: {train_size} ({train_size/total_samples*100:.1f}%)")
    if dev_size > 0:
        print(f"Dev samples: {dev_size} ({dev_size/total_samples*100:.1f}%)")
    print(f"Test samples: {test_size} ({test_size/total_samples*100:.1f}%)")

    # Split the data
    train_lines = data_lines[:train_size]
    if dev_size > 0:
        dev_lines = data_lines[train_size : train_size + dev_size]
        test_lines = data_lines[train_size + dev_size :]
    else:
        dev_lines = []
        test_lines = data_lines[train_size:]

    # Create output directory structure
    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    if dev_size > 0:
        dev_dir = output_dir / "dev"
        dev_dir.mkdir(parents=True, exist_ok=True)

    # Write train file
    train_output = train_dir / "data.tsv"
    print(f"Writing train data to: {train_output}")
    with open(train_output, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(train_lines)

    # Write dev file if needed
    if dev_size > 0:
        dev_output = dev_dir / "data.tsv"
        print(f"Writing dev data to: {dev_output}")
        with open(dev_output, "w", encoding="utf-8") as f:
            f.write(header)
            f.writelines(dev_lines)

    # Write test file
    test_output = test_dir / "data.tsv"
    print(f"Writing test data to: {test_output}")
    with open(test_output, "w", encoding="utf-8") as f:
        f.write(header)
        f.writelines(test_lines)

    print("Split completed successfully!")

    # Print statistics
    print("\n=== Summary ===")
    print(f"Input: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Train: {train_output} ({len(train_lines)} samples)")
    if dev_size > 0:
        print(f"Dev: {dev_output} ({len(dev_lines)} samples)")
    print(f"Test: {test_output} ({len(test_lines)} samples)")
    print(f"Random seed: {args.seed}")


if __name__ == "__main__":
    main()
