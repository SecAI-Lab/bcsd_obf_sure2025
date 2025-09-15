import os
import json
import argparse
import random
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def generate_heatmap_from_dataset(dataset_path, dataset_name, split_name):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    pair_counts = defaultdict(int)
    for pair in data:
        obf1 = pair["func1"]["obfuscator"]
        opt1 = pair["func1"]["optimizer_level"]
        obf2 = pair["func2"]["obfuscator"]
        opt2 = pair["func2"]["optimizer_level"]

        key1 = f"{obf1}_{opt1}"
        key2 = f"{obf2}_{opt2}"
        key = tuple(sorted([key1, key2]))
        pair_counts[key] += 1

    keys = sorted(set(k for pair in pair_counts.keys() for k in pair))
    heatmap_matrix = pd.DataFrame(0, index=keys, columns=keys)

    for (k1, k2), count in pair_counts.items():
        heatmap_matrix.loc[k1, k2] += count
        if k1 != k2:
            heatmap_matrix.loc[k2, k1] += count

    plt.figure(figsize=(12, 10))
    sns.heatmap(heatmap_matrix, annot=True, fmt="d", cmap="YlGnBu")
    plt.title(f"{split_name.capitalize()} Positive Pair Count Heatmap ({dataset_name})")
    plt.tight_layout()
    heatmap_path = os.path.join(os.path.dirname(dataset_path), f"heatmap_{split_name}_{dataset_name}.png")
    plt.savefig(heatmap_path)
    plt.close()
    print(f"Saved heatmap to: {heatmap_path}")

def check_duplicate_positive_pairs(data):
    duplicates = []
    for pair in data:
        if pair["label"] != 1:
            continue
        f1 = pair["func1"]
        f2 = pair["func2"]
        if (
            f1["function_name"] == f2["function_name"] and
            f1["obfuscator"] == f2["obfuscator"] and
            f1["optimizer_level"] == f2["optimizer_level"]
        ):
            duplicates.append(pair)
    if duplicates:
        print(f"[Warning] Found {len(duplicates)} duplicate positive pairs (same function/obf/opt).")
    else:
        print("[OK] No duplicate positive pairs found.")
    return duplicates


def check_invalid_negative_pairs(data):
    invalids = []
    for pair in data:
        if pair["label"] != 0:
            continue
        f1 = pair["func1"]
        f2 = pair["func2"]
        if f1["function_name"] == f2["function_name"]:
            invalids.append(pair)
    if invalids:
        print(f"[Warning] Found {len(invalids)} invalid negative pairs with same function name.")
    else:
        print("[OK] No invalid negative pairs found.")
    return invalids

if __name__ == "__main__":
    dataset_dir = os.path.abspath(os.path.join(os.getcwd(), "..", "dataset"))
    for split in ["train", "test"]:
        for dataset_name in ["tigress"]:
            dataset_path = os.path.join(dataset_dir, f"finetuning_{split}_dataset_{dataset_name}.json")
            if os.path.exists(dataset_path):
                generate_heatmap_from_dataset(dataset_path, dataset_name, split)

                with open(dataset_path, "r") as f:
                    data = json.load(f)
                check_duplicate_positive_pairs(data)
                check_invalid_negative_pairs(data)
            else:
                print(f"File not found: {dataset_path}")

