import os
import json
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing

import hparams as hp


def make_bpe_tokenizer(data_path, save_path):
    tokenizer = ByteLevelBPETokenizer(lowercase=True)

    tokenizer.train(
        files=data_path,
        vocab_size=hp.vocab_size,
        min_frequency=2,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"]
    )

    tokenizer._tokenizer.post_processor = BertProcessing(
        ("</s>", tokenizer.token_to_id("</s>")),
        ("<s>", tokenizer.token_to_id("<s>"))
    )
    tokenizer.enable_truncation(max_length=hp.enc_maxlen)
    tokenizer.save_model(save_path)
    print(f"[✔] BPE tokenizer saved to: {save_path}")


def main():
    bpe_data_path = "dataset/tokenizer_dataset.txt"
    bpe_tokenizer_save_path = "output/tokenizer"

    os.makedirs(bpe_tokenizer_save_path, exist_ok=True)

    vocab_path = os.path.join(bpe_tokenizer_save_path, "vocab.json")
    if not os.path.exists(vocab_path):
        print(f"[Info] vocab.json not found. Training BPE tokenizer...")
        os.makedirs(bpe_tokenizer_save_path, exist_ok=True)
        make_bpe_tokenizer(bpe_data_path, bpe_tokenizer_save_path)
    else:
        print(f"[Info] BPE tokenizer already exists at {vocab_path}. Skipping tokenizer making.")

if __name__ == "__main__":
    main()

