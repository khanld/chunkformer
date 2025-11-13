#!/usr/bin/env python3
# Copyright (c) 2024 ChunkFormer Authors

"""
Convert TSV format to list format.

Supports both ASR and classification tasks:
- ASR: key, wav, txt columns
- Classification: key, wav, and any number of label columns (e.g., gender_label, emotion_label)

Usage:
    python tsv_to_list.py <input_tsv_file>
"""

import json
import os
import sys

import pandas as pd


def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <input_tsv_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    base_dir = os.path.dirname(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    list_file = os.path.join(base_dir, f"{base_name}.list")
    wav_scp_file = os.path.join(base_dir, "wav.scp")

    # Read the .tsv file into a pandas DataFrame
    df = pd.read_csv(input_file, sep="\t", comment="#")
    df = df.dropna()

    print(f"Read {len(df)} samples from {input_file}")
    print(f"Columns: {list(df.columns)}")

    # Check if this is ASR or classification
    has_txt = "txt" in df.columns
    has_key = "key" in df.columns
    has_wav = "wav" in df.columns

    if not has_wav:
        print("Error: 'wav' column is required")
        sys.exit(1)

    # Generate the "key" column if not present
    if not has_key:
        df["key"] = df["wav"]
        print("Generated 'key' column from 'wav'")

    # Write the .list file (JSON format with all columns)
    with open(list_file, "w", encoding="utf-8") as list_out:
        for _, row in df.iterrows():
            row_dict = row.to_dict()
            list_out.write(json.dumps(row_dict, ensure_ascii=False) + "\n")

    # Write the text file (only for ASR with 'txt' column)
    if has_txt:
        text_file = os.path.join(base_dir, "text")
        df["txt"] = [str(txt).strip() for txt in df["txt"]]
        with open(text_file, "w", encoding="utf-8") as text_out:
            for _, row in df.iterrows():
                text_out.write(f"{row['key']} {row['txt']}\n")
        print(f"Output written to {list_file}, {text_file}, and {wav_scp_file}")
    else:
        print(f"Output written to {list_file} and {wav_scp_file}")
        print("(No 'txt' column found, skipped text file generation)")

    # Write the wav.scp file (key wav)
    with open(wav_scp_file, "w", encoding="utf-8") as wav_out:
        for _, row in df.iterrows():
            wav_out.write(f"{row['key']} {row['wav']}\n")


if __name__ == "__main__":
    main()
