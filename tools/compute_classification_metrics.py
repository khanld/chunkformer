#!/usr/bin/env python3
# Copyright (c) 2024 ChunkFormer Authors

"""
Compute classification metrics (accuracy, precision, recall, F1).
"""

import argparse
import json

import numpy as np
import yaml
from sklearn.metrics import classification_report, confusion_matrix


def get_args():
    parser = argparse.ArgumentParser(description="Compute classification metrics")
    parser.add_argument("--config", required=True, help="Training config YAML")
    parser.add_argument("--predictions", required=True, help="Predictions TSV file")
    parser.add_argument("--labels", required=True, help="Ground truth labels (data.list)")
    parser.add_argument("--output", required=True, help="Output metrics file")
    return parser.parse_args()


def load_ground_truth(labels_file, tasks):
    """Load ground truth labels."""
    gt = {}
    with open(labels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            key = data["key"]
            gt[key] = {}
            for task in tasks:
                label_key = f"{task}_label"
                if label_key in data:
                    label = data[label_key]
                    if isinstance(label, str):
                        label = int(label)
                    gt[key][task] = label

    return gt


def load_predictions(pred_file, tasks):
    """Load predictions."""
    predictions = {}
    with open(pred_file, "r", encoding="utf-8") as f:
        # Skip header if exists
        f.readline()

        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 1 + len(tasks):
                continue

            key = parts[0]
            predictions[key] = {}

            for i, task in enumerate(tasks):
                pred = int(parts[i + 1])
                predictions[key][task] = pred

    return predictions


def compute_metrics(gt, predictions, tasks):
    """Compute metrics for each task."""
    results = {}

    for task in tasks:
        y_true = []
        y_pred = []

        # Collect predictions and ground truth
        for key in gt:
            if key not in predictions:
                continue

            if task not in gt[key] or task not in predictions[key]:
                continue

            gt_label = gt[key][task]
            pred_label = predictions[key][task]

            if gt_label < 0:  # Skip invalid labels
                continue

            y_true.append(gt_label)
            y_pred.append(pred_label)

        if not y_true:
            print(f"Warning: No valid samples for task {task}")
            continue

        # Classification report (includes all metrics)
        class_report = classification_report(y_true, y_pred, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        results[task] = {
            "classification_report": class_report,
            "confusion_matrix": cm.tolist(),
            "num_samples": len(y_true),
        }

    return results


def print_and_save_results(results, tasks, output_file):
    """Print and save results."""
    lines = []

    lines.append("=" * 70)
    lines.append("Classification Metrics")
    lines.append("=" * 70)

    for task in tasks:
        if task not in results:
            continue

        r = results[task]
        lines.append(f"\n{task.upper()}:")
        lines.append(f"  Samples: {r['num_samples']}")

        lines.append("\n  Classification Report:")
        # Indent the classification report
        report_lines = r["classification_report"].strip().split("\n")
        for report_line in report_lines:
            lines.append(f"    {report_line}")

        lines.append("\n  Confusion Matrix:")
        cm = np.array(r["confusion_matrix"])
        for i, row in enumerate(cm):
            lines.append(f"    {i}: {row.tolist()}")

    lines.append("\n" + "=" * 70)

    # Print to console
    for line in lines:
        print(line)

    # Save to file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\nMetrics saved to: {output_file}")


def main():
    args = get_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    tasks = list(config.get("model_conf", {}).get("tasks", {}).keys())
    if not tasks:
        print("Error: No tasks defined in config")
        return

    print(f"Tasks: {tasks}")

    # Load data
    print(f"Loading ground truth from: {args.labels}")
    gt = load_ground_truth(args.labels, tasks)

    print(f"Loading predictions from: {args.predictions}")
    predictions = load_predictions(args.predictions, tasks)

    # Compute metrics
    print("\nComputing metrics...")
    results = compute_metrics(gt, predictions, tasks)

    # Print and save
    print_and_save_results(results, tasks, args.output)


if __name__ == "__main__":
    main()
