#!/bin/bash

# run from the root of the repository

export french_books_dir="data/french-pd-books"

# Download dataset from huggingface and save each book as txt file
pip install datasets tqdm
echo "Downloading French-PD-Books dataset"
python << EOF
import os
from tqdm import tqdm
from datasets import load_dataset

streamed_dataset = load_dataset("PleIAs/French-PD-Books", split='train', streaming=True)

out_dir = os.path.join(os.getenv("french_books_dir"), "french_books_txt")
os.makedirs(out_dir, exist_ok=True)

n_chars = 0
n_required = 500_000_000 * 4 * 10
progressbar = tqdm(total=n_required)
for row in iter(streamed_dataset):
    with open(os.path.join(out_dir, row["file_id"] + ".txt"), "w") as f:
        f.write(row["complete_text"])
    n_chars += row['character_count']
    progressbar.update(row['character_count'])
    if n_chars > n_required:
        break
EOF


# Download train test split
wget https://y5d6.c15.e2-3.dev/public-bucket/file_list_test_iid.npy -O $french_books_dir/file_list_test_iid.npy
wget https://y5d6.c15.e2-3.dev/public-bucket/file_list_train.npy -O $french_books_dir/file_list_train.npy


# Tokenize
# a bit awkward because we reuse the languini script for books3 with minimal changes. but it works.

# test set
python languini/dataset_lib/tokenise_languini_books.py \
    --split_npy_file $french_books_dir/file_list_test_iid.npy \
    --spm_model languini/vocabs/spm_models/french_books_16384.model
    --books3_dir $french_books_dir
    --books3_subdir french_books_txt
    --output_dir $french_books_dir

# train set
python languini/dataset_lib/tokenise_languini_books.py \
    --split_npy_file $french_books_dir/file_list_train.npy \
    --spm_model languini/vocabs/spm_models/french_books_16384.model
    --books3_dir $french_books_dir
    --books3_subdir french_books_txt
    --output_dir $french_books_dir