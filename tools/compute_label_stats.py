#!/usr/bin/env python3
# Copyright (c) 2024 ChunkFormer Authors

"""
Compute label statistics for classification tasks.
"""

import argparse
import json
import sys
from collections import Counter

import yaml


def get_args():
    parser = argparse.ArgumentParser(description="Compute label statistics")
    parser.add_argument("--config", required=True, help="Training config")
    parser.add_argument("--train_data", required=True, help="Training data list")
    parser.add_argument("--dev_data", help="Development data list")
    parser.add_argument("--test_data", help="Test data list")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    return parser.parse_args()


def load_data(data_file, tasks):
    """Load data and collect label statistics."""
    label_counts = {task: Counter() for task in tasks}
    total_samples = 0

    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_samples += 1

            for task in tasks:
                label_key = f"{task}_label"
                if label_key in data:
                    label = data[label_key]
                    if isinstance(label, str):
                        label = int(label)
                    if label >= 0:  # Valid label
                        label_counts[task][label] += 1

    return label_counts, total_samples


def print_statistics(name, label_counts, total_samples, tasks):
    """Print label statistics."""
    print(f"\n{'='*60}")
    print(f"{name} Statistics")
    print(f"{'='*60}")
    print(f"Total samples: {total_samples}")

    for task in tasks:
        print(f"\n{task.upper()}:")
        counts = label_counts[task]

        if not counts:
            print("  No valid labels found")
            continue

        total_labeled = sum(counts.values())
        print(f"  Labeled samples: {total_labeled}")
        print("  Label distribution:")

        for label in sorted(counts.keys()):
            count = counts[label]
            percentage = (count / total_labeled) * 100
            print(f"    Label {label}: {count:6d} ({percentage:5.2f}%)")


def save_label_mappings(output_dir, label_counts, tasks):
    """Save label mappings to files."""
    for task in tasks:
        counts = label_counts[task]
        if not counts:
            continue

        mapping_file = f"{output_dir}/{task}_labels.txt"
        with open(mapping_file, "w", encoding="utf-8") as f:
            for label in sorted(counts.keys()):
                # Format: label_id label_name (placeholder, user should edit)
                f.write(f"{label} class_{label}\n")

        print(f"\nCreated label mapping: {mapping_file}")
        print("  (Please edit this file to add meaningful class names)")


def main():
    args = get_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tasks = config.get("dataset_conf", {}).get("tasks", [])
    if not tasks:
        print("Error: No tasks defined in config")
        sys.exit(1)

    # Process training data
    print(f"Processing training data: {args.train_data}")
    train_counts, train_total = load_data(args.train_data, tasks)
    print_statistics("Training", train_counts, train_total, tasks)

    # Process dev data
    if args.dev_data:
        print(f"\nProcessing dev data: {args.dev_data}")
        dev_counts, dev_total = load_data(args.dev_data, tasks)
        print_statistics("Development", dev_counts, dev_total, tasks)

    # Process test data
    if args.test_data:
        print(f"\nProcessing test data: {args.test_data}")
        test_counts, test_total = load_data(args.test_data, tasks)
        print_statistics("Test", test_counts, test_total, tasks)

    # Save label mappings
    save_label_mappings(args.output_dir, train_counts, tasks)

    # Check for label consistency
    print(f"\n{'='*60}")
    print("Label Consistency Check")
    print(f"{'='*60}")

    for task in tasks:
        train_labels = set(train_counts[task].keys())

        if args.dev_data:
            dev_labels = set(dev_counts[task].keys())
            dev_only = dev_labels - train_labels
            if dev_only:
                print(f"\nWarning ({task}): Dev set has labels not in train: {dev_only}")

        if args.test_data:
            test_labels = set(test_counts[task].keys())
            test_only = test_labels - train_labels
            if test_only:
                print(f"\nWarning ({task}): Test set has labels not in train: {test_only}")

    print("\nStatistics computation complete!")


if __name__ == "__main__":
    main()
