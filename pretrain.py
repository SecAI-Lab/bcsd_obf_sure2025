import os
import json
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from multiprocessing import Pool, cpu_count

from transformers import (
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    RobertaConfig,
    RobertaTokenizerFast,
    RobertaForMaskedLM
)

import hparams as hp


class CustomDataset(Dataset):
    def __init__(self, text_path, tokenizer):
        self.tokenizer = tokenizer

        # Load all lines
        with open(text_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f]

        # Encode with multiprocessing
        print(f"[INFO] Encoding {len(lines)} lines with multiprocessing...")
        with Pool(processes=cpu_count()) as pool:
            self.examples = list(tqdm(pool.imap(self._encode_line, lines), total=len(lines), desc="Tokenizing"))

    def _encode_line(self, line):
        encoded = self.tokenizer.encode_plus(
            line,
            max_length=hp.enc_maxlen,
            truncation=True,
            padding='max_length'
        )
        return encoded.input_ids

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx])



def train_roberta_model(tokenizer_path, train_data_path, save_path):
    # Config
    config = RobertaConfig(
        vocab_size=hp.vocab_size,
        hidden_size=hp.hidden_size,
        max_position_embeddings=hp.enc_maxlen + 2,
        num_attention_heads=hp.num_attention_heads,
        num_hidden_layers=hp.num_hidden_layers,
        output_hidden_states=True,
        output_attentions=True
    )

    model = RobertaForMaskedLM(config=config)

    tokenizer = RobertaTokenizerFast.from_pretrained(
        tokenizer_path,
        model_max_length=hp.enc_maxlen,
        truncation=True,
        padding='max_length'
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=hp.mlm_prob
    )

    train_dataset = CustomDataset(train_data_path, tokenizer)

    training_args = TrainingArguments(
        output_dir=save_path,
        overwrite_output_dir=True,
        num_train_epochs=hp.pretrain_epochs,
        per_device_train_batch_size=hp.pretrain_batch_size,
        learning_rate=hp.lr,
        weight_decay=hp.adam_weight_decay_rate,
        adam_beta1=hp.adam_beta1,
        adam_beta2=hp.adam_beta2,
        warmup_ratio=0.1,
        warmup_steps=hp.pretrain_epochs * len(train_dataset),
        max_grad_norm=hp.max_grad_norm,
        adam_epsilon=hp.epsilon,
        save_steps=hp.save_steps,
        eval_steps=hp.save_steps,
        logging_steps=hp.save_steps,
        save_total_limit=1,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    print("Training RoBERTa model...")
    trainer.train()

    print(f"Saving model to: {save_path}")
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)


def main():
    parser = argparse.ArgumentParser(description="Pre-train RoBERTa with BPE tokenizer")
    parser.add_argument('--dataset_name', type=str, required=True, choices=['ollvm', 'tigress'],
                        help="Choose dataset: 'ollvm' or 'tigress'")
    parser.add_argument('--device', type=str, default="0",
                        help='GPU device id to use (e.g., "0", "1")')

    args = parser.parse_args()

    # Device environment setting
    if args.device == "0":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

    if args.dataset_name == "ollvm":
        pretrain_model_save_path = f"output/ollvm/pretrained"
        pretrain_data_path = "dataset/pretrain_train_dataset_ollvm.txt"
    elif args.dataset_name == "tigress":
        pretrain_model_save_path = f"output/tigress/pretrained"
        pretrain_data_path = "dataset/pretrain_train_dataset_tigress.txt"
    else:
        ValueError(f"Unsupported dataset name: {args.dataset_name}. Please choose either 'ollvm' or 'tigress'.")

    os.makedirs(pretrain_model_save_path, exist_ok=True)

    bpe_tokenizer_save_path = f"output/tokenizer"

    print(f"[INFO] Loaded dataset path: {pretrain_data_path}")
    print(f"[INFO] Model will be saved to: {pretrain_model_save_path}")
    print(f"[INFO] Using tokenizer from: {bpe_tokenizer_save_path}")

    train_roberta_model(
        tokenizer_path=bpe_tokenizer_save_path,
        train_data_path=pretrain_data_path,
        save_path=pretrain_model_save_path
    )


if __name__ == "__main__":
    main()

