import os
import csv
from datasets import load_dataset

SYNTH_OCR_DATA_PATH = os.path.abspath(os.getenv("SYNTH_OCR_DATA_PATH", "../synthtiger_gen/results"))
GT_FILE = os.path.join(SYNTH_OCR_DATA_PATH, "gt_1.txt")

dataset = load_dataset(
    "text",  # Load as raw text
    data_files={"train": GT_FILE},
    split="train",
    streaming=True
)

def process_line(example):
    parts = example["text"].split("\t", 1)  # Split into 2 parts
    return {"file_path": parts[0], "text": parts[1] if len(parts) > 1 else ""}

dataset = dataset.map(process_line)

for i, row in enumerate(dataset):
    if not isinstance(row, dict) or "file_path" not in row or "text" not in row:
        print(f"❌ Issue at line {i}: {row}")  # Row structure is broken
        break
    
    if not row["file_path"].endswith((".jpg", ".png", ".jpeg")):
        print(f"⚠️ Unexpected file path at line {i}: {row['file_path']}")
        break

    if "\n" in row["text"] or "\r" in row["text"]:
        print(f"⚠️ Newline detected in text at line {i}: {row['text']}")
        break

    if "\t" in row["text"]:  # There should be only one tab per line
        print(f"⚠️ Extra tab detected in text at line {i}: {row['text']}")
        break

    if row["file_path"].count("\t") > 0:
        print(f"⚠️ Extra tab detected in file path at line {i}: {row['file_path']}")
        break

    print(row)