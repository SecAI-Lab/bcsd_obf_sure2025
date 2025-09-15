import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast, RobertaModel
import warnings
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_curve, auc as calc_auc

import hparams as hp
import Adam_opti as op
from finetune import SimDataset, SimilarityModel, SimilarityTrainer, collate_sim, Paths

warnings.filterwarnings("ignore", message="To copy construct from a tensor")


BASE_DIR = "./sure2025_ver2"
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATHS = {
    "BinShot_OLLVM": os.path.join(BASE_DIR, "output", "ollvm", "finetuned"),
    "BinShot_Tigress": os.path.join(BASE_DIR, "output", "tigress", "finetuned"),
}

TEST_CASES = [
    ("ollvm_benign_clang", "BinShot_OLLVM", "ollvm_benign_clang_BinShot_OLLVM_results.json"),
    ("tigress_benign_gcc", "BinShot_Tigress", "tigress_benign_gcc_BinShot_Tigress_results.json"),
    ("benign_benign_clang", "BinShot_OLLVM", "benign_benign_clang_BinShot_OLLVM_results.json"),
    ("benign_benign_gcc", "BinShot_Tigress", "benign_benign_gcc_BinShot_Tigress_results.json"),
    ("tigress_benign_gcc", "BinShot_OLLVM", "tigress_benign_gcc_BinShot_OLLVM_results.json"),
    ("ollvm_benign_clang", "BinShot_Tigress", "ollvm_benign_clang_BinShot_Tigress_results.json"),
]

DATASET_FILES = {
    "ollvm_benign_clang": [
        "OLLVM_sub_none_clang.json",
        "OLLVM_fla_none_clang.json",
        "OLLVM_bcf_none_clang.json",
        "OLLVM_all_none_clang.json"
    ],
    "tigress_benign_gcc": [
        "Tigress_EncodeArithmetic_none_gcc.json",
        "Tigress_AddOpaque_none_gcc.json",
        "Tigress_EncodeBranches_none_gcc.json",
        "Tigress_Flatten_none_gcc.json",
        "Tigress_Virtualize_none_gcc.json"
    ],
    "benign_benign_clang": ["OLLVM_benign_benign_clang.json"],
    "benign_benign_gcc": ["Tigress_benign_benign_gcc.json"]
}



def load_similarity_model(model_dir, device):
    encoder_path = os.path.join(model_dir, "roberta_ep9")
    sim_model_path = os.path.join(model_dir, "sim_ep9.pt")

    print(f"[+] Loading model from {model_dir}")
    roberta = RobertaModel.from_pretrained(encoder_path)
    roberta.eval()

    model = SimilarityModel(roberta=roberta, device=device).to(device)
    checkpoint = torch.load(sim_model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    tokenizer = RobertaTokenizerFast.from_pretrained(encoder_path)
    return model, tokenizer

def compute_metric(pred, label):
    precision, recall, f1, _ = precision_recall_fscore_support(label, pred, average='binary')
    acc = accuracy_score(label, pred)
    fpr, tpr, _ = roc_curve(label, pred, pos_label=1)
    auc_value = calc_auc(fpr, tpr)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'auc': auc_value
    }


def inference_and_save(all_data, all_preds, result_save_path):
    print(f"[INFO] Saving result to: {result_save_path}")
    output_data = []
    for item, pred in zip(all_data, all_preds):
        item['func1'].pop('asm_code', None)
        item['func2'].pop('asm_code', None)
        item['pred'] = int(pred)
        output_data.append(item)
    with open(result_save_path, "w") as f:
        json.dump(output_data, f, indent=2)


def run_all_inference(device="cuda:0"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    for dataset_name, model_key, result_file in TEST_CASES:
        print("=" * 80)
        print(f"Processing: Dataset [{dataset_name}] with Model [{model_key}]")

        model_dir = MODEL_PATHS[model_key]
        model, tokenizer = load_similarity_model(model_dir, device)

        all_data, all_preds, all_labels = [], [], []

        for subset_file in DATASET_FILES[dataset_name]:
            subset_path = os.path.join(DATASET_DIR, subset_file)
            with open(subset_path, 'r') as f:
                raw_data = json.load(f)

            test_dataset = SimDataset(subset_path, tokenizer)
            test_loader = DataLoader(test_dataset, batch_size=hp.finetune_batch_size, shuffle=False, collate_fn=collate_sim)

            trainer = SimilarityTrainer(device, None, None, test_loader, None, tokenizer)
            trainer.model = model
            trainer.set_test_data(test_loader)

            preds = trainer.predict()
            labels = np.array([item['label'] for item in raw_data])
            metric = compute_metric(preds, labels)
            print(f"[{subset_file}] --> Acc: {metric['accuracy']:.4f}, Prec: {metric['precision']:.4f}, "
                  f"Recall: {metric['recall']:.4f},  F1: {metric['f1']:.4f}, AUC: {metric['auc']:.4f}")

            all_data.extend(raw_data)
            all_preds.extend(preds)
            all_labels.extend(labels)

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        metric = compute_metric(all_preds, all_labels)
        print(f"[Total] --> Acc: {metric['accuracy']:.4f}, Prec: {metric['precision']:.4f}, "
              f"Recall: {metric['recall']:.4f}, F1: {metric['f1']:.4f}, AUC: {metric['auc']:.4f}")

        result_save_path = os.path.join(OUTPUT_DIR, result_file)
        inference_and_save(all_data, all_preds, result_save_path)

        print("=" * 80)


if __name__ == "__main__":
    run_all_inference()
