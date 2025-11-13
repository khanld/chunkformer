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
and automatically creates a label_mapping.json file.

Example:
    Input TSV:
        key     wav     gender_label    emotion_label
        utt1    a.wav   male           happy
        utt2    b.wav   female         sad

    Output TSV (always data.tsv):
        key     wav     gender_label    emotion_label
        utt1    a.wav   0               0
        utt2    b.wav   1               1

    Note: If input file is named "data.tsv", it will be renamed to "data_original.tsv"

    Label mappings (label_mapping.json, automatically created):
        {
            "gender": {
                "0": "male",
                "1": "female"
            },
            "emotion": {
                "0": "happy",
                "1": "sad"
            }
        }
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
        "--tasks",
        "-t",
        nargs="+",
        help="List of task names (e.g., gender emotion region). "
        "If not specified, will auto-detect from column names ending with '_label'",
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


def load_label_mapping(label_mapping_file):
    """Load existing label mapping from JSON file."""
    if not os.path.exists(label_mapping_file):
        return None

    with open(label_mapping_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_label_mapping(label_mapping_file, all_mappings):
    """Save all task label mappings to a single JSON file."""
    label_dir = os.path.dirname(label_mapping_file)
    os.makedirs(label_dir, exist_ok=True)

    with open(label_mapping_file, "w", encoding="utf-8") as f:
        json.dump(all_mappings, f, indent=2, ensure_ascii=False)


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

    # Create mapping as id: label (reversed from before)
    mapping = {str(idx): label for idx, label in enumerate(sorted_labels)}

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

            # Now label_mappings[task] is {id: label}, so we need to reverse lookup
            # Create reverse mapping: label -> id
            reverse_mapping = {
                label: int(label_id) for label_id, label in label_mappings[task].items()
            }

            if text_label not in reverse_mapping:
                missing_labels[task].add(text_label)
                continue

            new_row[label_key] = str(reverse_mapping[text_label])

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

    # Generate output filenames based on input
    input_dir = os.path.dirname(args.input) or "."
    input_basename = os.path.basename(args.input)

    # Output file is always named "data.tsv"
    output_file = os.path.join(input_dir, "data.tsv")
    label_mapping_file = os.path.join(input_dir, "label_mapping.json")

    # If input is already "data.tsv", rename it to "data_original.tsv"
    if input_basename == "data.tsv":
        original_input_file = os.path.join(input_dir, "data_original.tsv")
        print(f"Input file is 'data.tsv', renaming to: {original_input_file}")
        os.rename(args.input, original_input_file)
        args.input = original_input_file

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

    # Create label mappings from data
    all_label_mappings = {}

    print("\nCreating label mappings for all tasks...")
    for task in tasks:
        print(f"  Task '{task}'...")
        all_label_mappings[task] = create_label_mapping(rows, task)
        labels_list = [
            f"{label_id}: {label}" for label_id, label in all_label_mappings[task].items()
        ]
        print(f"    Found {len(all_label_mappings[task])} unique labels: {labels_list}")

    save_label_mapping(label_mapping_file, all_label_mappings)
    print(f"\nSaved all label mappings to: {label_mapping_file}")

    # Convert labels
    print("\nConverting labels...")
    converted_rows = convert_labels(rows, tasks, all_label_mappings)

    # Write output (always named "data.tsv")
    print(f"Writing output file: {output_file}")
    write_tsv(output_file, header, converted_rows)
    print(f"  Wrote {len(converted_rows)} rows")

    print("\nDone!")
    print(f"  Output TSV: {output_file}")
    print(f"  Label mapping: {label_mapping_file}")


if __name__ == "__main__":
    main()
