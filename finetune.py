import os
import json
import argparse
import torch
import numpy as np
import random
import re
import traceback
import sys
import warnings
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, RobertaModel

import hparams as hp
import Adam_opti as op


def pad1d(x, max_len):
    return np.pad(x, (0, max_len - len(x)), mode='constant')


def collate_sim(batch):
    f1_input_lens = [len(x[0]) for x in batch]
    f2_input_lens = [len(x[1]) for x in batch]
    max_x_len = max(f1_input_lens + f2_input_lens)

    f1_instrs = [pad1d(x[0], max_x_len) for x in batch]
    f2_instrs = [pad1d(x[1], max_x_len) for x in batch]

    f1_instrs = np.stack(f1_instrs)
    f2_instrs = np.stack(f2_instrs)

    labels = torch.tensor([x[2] for x in batch]).float()
    lines = [x[3] for x in batch]

    return {
        "f1_input": f1_instrs,
        "f2_input": f2_instrs,
        "label": labels,
        "line": lines
    }

class SimDataset:
    def __init__(self, json_path, tokenizer, encoding="utf-8"):
        self.tokenizer = tokenizer
        self.corpus = []

        with open(json_path, "r", encoding=encoding) as f:
            data = json.load(f)

        for entry in tqdm(data, desc="Loading JSON Dataset"):
            f1 = entry["func1"]["asm_code"]
            f2 = entry["func2"]["asm_code"]
            label = int(entry["label"])

            self.corpus.append((f1, f2, label, json.dumps(entry)))

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        f1, f2, label, line = self.corpus[idx]
        f1_ids = self.tokenizer(f1, truncation=True, padding='max_length', max_length=hp.enc_maxlen).input_ids
        f2_ids = self.tokenizer(f2, truncation=True, padding='max_length', max_length=hp.enc_maxlen).input_ids
        return f1_ids, f2_ids, label, line

class SimilarityModel(nn.Module):
    def __init__(self, roberta, device):
        super().__init__()
        self.device = device
        self.roberta = roberta
        self.l2_dist = nn.MSELoss(reduction="none")
        self.linear = nn.Linear(hp.num_hidden, 1)

    def forward(self, f1_input, f2_input=None):
        f1_input = f1_input.clone().detach().to(self.device)
        out1 = self.roberta(f1_input).last_hidden_state[:, 0, :]
        f2_input = f2_input.clone().detach().to(self.device)
        out2 = self.roberta(f2_input).last_hidden_state[:, 0, :]

        dist = self.l2_dist(out1, out2)
        logits = self.linear(dist).squeeze(1)
        return logits


class SimilarityTrainer:
    def __init__(self, device, roberta, train_dataloader, test_dataloader, path, tokenizer, log_freq=hp.log_freq):
        self.device = device
        self.roberta = roberta

        if self.roberta is not None:
           self.model = SimilarityModel(roberta, self.device).to(self.device)
        else:
            self.model = None


        self.tokenizer = tokenizer
        self.train_data = train_dataloader
        self.test_data = test_dataloader
        self.path = path
        self.step = 0
        self.log_freq = log_freq
        self.criterion = nn.BCEWithLogitsLoss()

                
        if train_dataloader is not None:
            total_steps = hp.finetune_epochs * len(train_dataloader)
            self.optimizer = op.optim4GPU(self.model, total_steps)
        else:
            self.optimizer = None

    def train(self):
        for epoch in range(hp.finetune_epochs):
            self.model.train()
            total_loss, preds, labels = 0, [], []

            for i, batch in enumerate(tqdm(self.train_data, desc=f"Epoch {epoch} [Train]")):
                self.step += 1
                batch = {k: torch.tensor(v).to(self.device) if k != 'line' else v for k, v in batch.items()}
                logits = self.model(batch['f1_input'], batch['f2_input'])

                loss = self.criterion(logits, batch['label'])
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                total_loss += loss.item()
                preds.extend(torch.round(torch.sigmoid(logits)).detach().cpu().numpy())
                labels.extend(batch["label"].detach().cpu().numpy())

                if i % self.log_freq == 0:
                    metrics = op.compute_prediction_metric(np.array(preds), np.array(labels), avg='binary')
                    print(f"[Train Step {self.step}] Loss: {total_loss / (i + 1):.4f}, "
                          f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

            metrics = op.compute_prediction_metric(np.array(preds), np.array(labels), avg='binary')
            print(f"[Epoch {epoch}] Avg Loss: {total_loss / len(self.train_data):.4f}, "
                  f"Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

            self.save_model(epoch)

    def test(self):
        self.model.eval()
        preds, labels = [], []

        with torch.no_grad():
            for batch in tqdm(self.test_data, desc="[Test]"):
                batch = {k: torch.tensor(v).to(self.device) if k != 'line' else v for k, v in batch.items()}
                logits = self.model(batch['f1_input'], batch['f2_input'])

                preds.extend(torch.round(torch.sigmoid(logits)).cpu().numpy())
                labels.extend(batch["label"].cpu().numpy())

        metrics = op.compute_prediction_metric(np.array(preds), np.array(labels), avg='binary')
        print(f"[Test Result] Acc: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, "
              f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, AUC: {metrics['auc']:.4f}")

    def predict(self):
        self.model.eval()
        preds = []

        with torch.no_grad():
            for batch in tqdm(self.test_data, desc="[Predict]"):
                batch = {k: torch.tensor(v).to(self.device) if k != 'line' else v for k, v in batch.items()}
                logits = self.model(batch['f1_input'], batch['f2_input'])
                preds.extend(torch.round(torch.sigmoid(logits)).cpu().numpy())

        return preds

    def save_model(self, epoch):
        model_path = os.path.join(self.path.sim_path, f"sim_ep{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, model_path)
        print(f"[Model Saved] {model_path}")

        roberta_path = os.path.join(self.path.sim_path, f"roberta_ep{epoch}")
        self.model.roberta.save_pretrained(roberta_path)
        self.tokenizer.save_pretrained(roberta_path)
        print(f"[RoBERTa Saved] {roberta_path}")

    def set_test_data(self, test_loader):
        self.test_data = test_loader
        
class Paths:
    def __init__(self, base_path):
        self.output_path = base_path
        self.sim_path = os.path.join(base_path)
        os.makedirs(self.sim_path, exist_ok=True)

def run_finetuning(pretrain_path, train_corpus, test_corpus, bcsd_save_path, seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.makedirs(bcsd_save_path, exist_ok=True)

    tokenizer = RobertaTokenizerFast.from_pretrained(pretrain_path)
    print(f"max_token: {tokenizer.model_max_length}")

    train_dataset = SimDataset(train_corpus, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=collate_sim)

    print("[+] Loading pretrained model...")
    roberta = RobertaModel.from_pretrained(pretrain_path)
    roberta.eval()

    print("[+] Starting training...")
    paths = Paths(bcsd_save_path)
    trainer = SimilarityTrainer(device, roberta, train_loader, None, paths, tokenizer)
    trainer.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Fine-tune RoBERTa for BCSD")
    parser.add_argument('--pretrain_path', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset_name', type=str, required=True, choices=['ollvm', 'tigress'],
                        help="Choose dataset: 'ollvm' or 'tigress'")

    parser.add_argument('--device', type=str, default="1",
                        help='GPU device id to use (e.g., "0", "1")')

    args = parser.parse_args()
    pretrain_path = f"output/{args.dataset_name}/pretrained"
    finetune_train_data = f"dataset/finetuning_train_dataset_{args.dataset_name}_100.json"
    finetune_save_path = f"output/{args.dataset_name}/finetuned"
    device = torch.device(f"cuda:{args.device}")

    print(f"Train BCSD Model [{args.dataset_name}]")
    print(f"[INFO] Loaded dataset path: {finetune_train_data}")
    print(f"[INFO] Model will be saved to: {finetune_save_path}")

    warnings.filterwarnings("ignore", message="To copy construct from a tensor")
    run_finetuning(pretrain_path, finetune_train_data, None, finetune_save_path, args.seed, device)
