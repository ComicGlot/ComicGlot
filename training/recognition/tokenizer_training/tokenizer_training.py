import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from transformers import AutoTokenizer
from pathlib import Path

#corpus_file = "E:/ko.txt"
corpus_file = "./OCR/OCR_training/ko_corpus/full_corpus_nltk.txt"

# ------------------------------------------------------
# def read_corpus_in_chunks(filepath, start, end):
#     with open(filepath, "r", encoding="utf-8") as file:
#         for i, line in enumerate(file):
#             if start <= i < end:
#                 yield line.strip()

# def train_on_chunk(start, end, output_dir):
#     print(f"Processing lines {start} to {end}")
#     tokenizer = AutoTokenizer.from_pretrained("team-lucid/trocr-small-korean")
#     tokenizer = tokenizer.train_new_from_iterator(
#         read_corpus_in_chunks(corpus_file, start, end),
#         vocab_size=52000
#     )
#     tokenizer.save_pretrained(f"{output_dir}/chunk_{start}_{end}")

# state_file = Path("training_state.txt")
# if state_file.exists():
#     with state_file.open("r") as f:
#         last_processed = int(f.read().strip())
# else:
#     last_processed = 0

# chunk_size = 1_000_000
# num_lines = sum(1 for _ in open(corpus_file, "r", encoding="utf-8"))
# total_chunks = (num_lines + chunk_size - 1) // chunk_size

# for chunk_start in range(last_processed, total_chunks * chunk_size, chunk_size):
#     chunk_end = min(chunk_start + chunk_size, num_lines)
#     print(f"Processing lines {chunk_start} to {chunk_end}")

#     train_on_chunk(chunk_start, chunk_end, "./custom_trocr_tokenizer")

#     with state_file.open("w") as f:
#         f.write(str(chunk_start))
# ------------------------------------------------------

def read_corpus(filepath):
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            yield line.strip()

original_tokenizer = AutoTokenizer.from_pretrained("team-lucid/trocr-small-korean")

new_tokenizer = original_tokenizer.train_new_from_iterator(
    read_corpus(corpus_file), vocab_size=52000
)

new_tokenizer.save_pretrained("./ko_trocr_nltkC_tokenizer")
