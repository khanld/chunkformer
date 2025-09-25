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
    text_file = os.path.join(base_dir, "text")
    wav_scp_file = os.path.join(base_dir, "wav.scp")

    # Read the .tsv file into a pandas DataFrame
    df = pd.read_csv(input_file, sep="\t")
    df = df.dropna()

    # Generate the "key" and "wav" columns
    df["key"] = df["wav"]

    # Write the .list file
    with open(list_file, "w", encoding="utf-8") as list_out:
        for _, row in df.iterrows():
            row_dict = {"key": row["key"], "wav": row["wav"], "txt": row["txt"]}
            list_out.write(json.dumps(row_dict, ensure_ascii=False) + "\n")

    # Write the text file (key txt)
    df["txt"] = [str(txt).strip() for txt in df["txt"]]
    with open(text_file, "w", encoding="utf-8") as text_out:
        for _, row in df.iterrows():
            text_out.write(f"{row['key']} {row['txt']}\n")

    # Write the wav.scp file (key wav)
    with open(wav_scp_file, "w", encoding="utf-8") as wav_out:
        for _, row in df.iterrows():
            wav_out.write(f"{row['key']} {row['wav']}\n")
    print(f"Output written to {list_file}, {text_file}, and {wav_scp_file}")


if __name__ == "__main__":
    main()
