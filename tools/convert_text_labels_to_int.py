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

"""Convert text classification labels to integer labels.

This script converts classification labels from text format to integer format
and creates label mapping files.

Example:
    Input TSV:
        key     wav     gender_label    emotion_label
        utt1    a.wav   male           happy
        utt2    b.wav   female         sad

    Output TSV:
        key     wav     gender_label    emotion_label
        utt1    a.wav   0               0
        utt2    b.wav   1               1

    Label mappings (gender.txt):
        male 0
        female 1

    Label mappings (emotion.txt):
        happy 0
        sad 1
"""

import argparse
import json
import os
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert text classification labels to integer labels"
    )
    parser.add_argument("--input", "-i", required=True, help="Input TSV file with text labels")
    parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output TSV file with integer labels",
    )
    parser.add_argument(
        "--label-dir",
        "-l",
        required=True,
        help="Output directory for label mapping files",
    )
    parser.add_argument(
        "--tasks",
        "-t",
        nargs="+",
        help="List of task names (e.g., gender emotion region). "
        "If not specified, will auto-detect from column names ending with '_label'",
    )
    parser.add_argument(
        "--auto-create",
        "-a",
        action="store_true",
        help="Automatically create label mappings from data. "
        "If False, will use existing label files in label-dir",
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=["txt", "json"],
        default="txt",
        help="Label mapping file format (default: txt)",
    )
    return parser.parse_args()


def read_tsv(input_file):
    """Read TSV file and return header and rows."""
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if not lines:
        raise ValueError(f"Empty input file: {input_file}")

    # Parse header
    header = lines[0].strip().split("\t")

    # Parse data rows
    rows = []
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) != len(header):
            print(f"Warning: Skipping malformed line: {line}")
            continue
        rows.append(dict(zip(header, parts)))

    return header, rows


def detect_tasks(header):
    """Detect task names from header columns ending with '_label'."""
    tasks = []
    for col in header:
        if col.endswith("_label"):
            task_name = col[:-6]  # Remove '_label' suffix
            tasks.append(task_name)
    return tasks


def load_label_mapping(label_file, format="txt"):
    """Load existing label mapping from file."""
    if not os.path.exists(label_file):
        return None

    if format == "txt":
        mapping = {}
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 2:
                    text_label, int_label = parts
                    mapping[text_label] = int(int_label)
        return mapping
    elif format == "json":
        with open(label_file, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown format: {format}")


def save_label_mapping(label_file, mapping, format="txt"):
    """Save label mapping to file."""
    label_dir = os.path.dirname(label_file)
    if label_dir:  # Only create dir if there's a directory path
        os.makedirs(label_dir, exist_ok=True)

    if format == "txt":
        with open(label_file, "w", encoding="utf-8") as f:
            for text_label in sorted(mapping.keys(), key=lambda x: mapping[x]):
                f.write(f"{text_label} {mapping[text_label]}\n")
    elif format == "json":
        with open(label_file, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
    else:
        raise ValueError(f"Unknown format: {format}")


def create_label_mapping(rows, task):
    """Create label mapping from data by collecting all unique labels."""
    label_key = f"{task}_label"
    unique_labels = set()

    for row in rows:
        if label_key in row:
            label = row[label_key].strip()
            if label:  # Skip empty labels
                unique_labels.add(label)

    # Sort labels alphabetically for consistency
    sorted_labels = sorted(unique_labels)

    # Create mapping
    mapping = {label: idx for idx, label in enumerate(sorted_labels)}

    return mapping


def convert_labels(rows, tasks, label_mappings):
    """Convert text labels to integer labels."""
    converted_rows = []
    missing_labels = defaultdict(set)

    for row in rows:
        new_row = row.copy()
        for task in tasks:
            label_key = f"{task}_label"
            if label_key not in row:
                print(f"Warning: Missing {label_key} in row with key {row.get('key', 'unknown')}")
                continue

            text_label = row[label_key].strip()
            if not text_label:
                print(f"Warning: Empty {label_key} in row with key {row.get('key', 'unknown')}")
                continue

            if text_label not in label_mappings[task]:
                missing_labels[task].add(text_label)
                continue

            new_row[label_key] = str(label_mappings[task][text_label])

        converted_rows.append(new_row)

    # Report missing labels
    if missing_labels:
        print("\nWarning: Some text labels not found in mappings:")
        for task, labels in missing_labels.items():
            print(f"  Task '{task}': {', '.join(sorted(labels))}")

    return converted_rows


def write_tsv(output_file, header, rows):
    """Write TSV file."""
    output_dir = os.path.dirname(output_file)
    if output_dir:  # Only create dir if there's a directory path
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        # Write header
        f.write("\t".join(header) + "\n")

        # Write data rows
        for row in rows:
            values = [row.get(col, "") for col in header]
            f.write("\t".join(values) + "\n")


def main():
    args = parse_args()

    print(f"Reading input file: {args.input}")
    header, rows = read_tsv(args.input)
    print(f"  Found {len(rows)} rows")

    # Detect or use specified tasks
    if args.tasks:
        tasks = args.tasks
    else:
        tasks = detect_tasks(header)

    if not tasks:
        raise ValueError(
            "No tasks found. Please specify --tasks or ensure columns end with '_label'"
        )

    print(f"Tasks to process: {', '.join(tasks)}")

    # Load or create label mappings
    label_mappings = {}
    os.makedirs(args.label_dir, exist_ok=True)

    for task in tasks:
        label_file = os.path.join(args.label_dir, f"{task}.{args.format}")

        if args.auto_create:
            print(f"\nCreating label mapping for task '{task}'...")
            label_mappings[task] = create_label_mapping(rows, task)
            save_label_mapping(label_file, label_mappings[task], args.format)
            print(f"  Saved mapping to: {label_file}")
            labels_list = list(label_mappings[task].keys())
            print(f"  Found {len(label_mappings[task])} unique labels: {labels_list}")
        else:
            print(f"\nLoading label mapping for task '{task}' from {label_file}...")
            mapping = load_label_mapping(label_file, args.format)
            if mapping is None:
                raise ValueError(
                    f"Label mapping file not found: {label_file}. "
                    "Use --auto-create to create mappings from data."
                )
            label_mappings[task] = mapping
            print(f"  Loaded {len(mapping)} labels")

    # Convert labels
    print("\nConverting labels...")
    converted_rows = convert_labels(rows, tasks, label_mappings)

    # Write output
    print(f"Writing output file: {args.output}")
    write_tsv(args.output, header, converted_rows)
    print(f"  Wrote {len(converted_rows)} rows")

    print("\nDone!")


if __name__ == "__main__":
    main()
