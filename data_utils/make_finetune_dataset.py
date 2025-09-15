import os
import json
import argparse
import random
from tqdm import tqdm
from collections import defaultdict
from itertools import combinations

random.seed(42)

# Constants
LLVM_OBFUSCATORS = {"none", "bcf", "fla", "sub", "all"}
TIGRESS_OBFUSCATORS = {
    "AddOpaque", "EncodeArithmetic", "EncodeBranches",
    "Flatten", "Virtualize", "none"
}

# Utilities
def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def group_by_fn_strict(items):
    fn_map = defaultdict(list)
    for item in items:
        key = (item["corpus"], item["binary_name"], item["function_name"])
        fn_map[key].append(item)
    return fn_map

def build_positive_pairs(fn_map, dataset_name, data_info, reduce_ratio=0.5):        
    positives = []
    pos_stats = defaultdict(list)

    for items in tqdm(fn_map.values(), desc=f"Building positive pairs: [{data_info}]"):
        local_pairs = []
        for a, b in combinations(items, 2):
            if (a["optimizer_level"], a["obfuscator"]) == (b["optimizer_level"], b["obfuscator"]):
                continue
            if dataset_name == "ollvm" and {a["optimizer_level"], b["optimizer_level"]} == {"O2", "O3"}:
                continue
            key = tuple(sorted([
                (a["optimizer_level"], a["obfuscator"]),
                (b["optimizer_level"], b["obfuscator"])
            ]))
            local_pairs.append((key, {"func1": a, "func2": b, "label": 1}))

        if dataset_name == "ollvm" and reduce_ratio < 1.0 and len(local_pairs) > 1:
            sample_size = max(1, int(len(local_pairs) * reduce_ratio))
            local_pairs = random.sample(local_pairs, sample_size)

        for key, pair in local_pairs:
            pos_stats[key].append(pair)
            positives.append(pair)

    return positives, pos_stats


def build_negative_pairs(fn_map, pos_stats, dataset_name, data_info):
    negatives = []
    all_keys = list(fn_map.keys())
    pbar = tqdm(total=sum(len(v) for v in pos_stats.values()), desc=f"Generating negatives paris: [{data_info}]")

    for pair_key, pos_list in pos_stats.items():
        count = len(pos_list)
        added = 0
        attempts = 0
        max_attempts = count * 10

        while added < count and attempts < max_attempts:
            k1, k2 = random.sample(all_keys, 2)
            fn1 = k1[2]
            fn2 = k2[2]
            if fn1 == fn2:
                attempts += 1
                continue
            a = random.choice(fn_map[k1])
            b = random.choice(fn_map[k2])

            negatives.append({"func1": a, "func2": b, "label": 0})
            added += 1
            pbar.update(1)

    pbar.close()
    return negatives

def generate_finetune_dataset(data_info, dataset_name, data, output_dir):
    if dataset_name == "ollvm":
        samples = [item for item in data if item["obfuscator"] in LLVM_OBFUSCATORS]
    else:
        samples = [item for item in data if item["obfuscator"] in TIGRESS_OBFUSCATORS]

    fn_map = group_by_fn_strict(samples)
    positives, pos_stats = build_positive_pairs(fn_map, dataset_name, data_info, reduce_ratio=0.5)
    negatives = build_negative_pairs(fn_map, pos_stats, dataset_name, data_info)

    full_dataset = positives + negatives
    random.shuffle(full_dataset)

    output_path = os.path.join(output_dir, f"finetuning_{data_info}_dataset_{dataset_name}.json")
    save_json(output_path, full_dataset)

    print(f"[{data_info}] Positive: {len(positives)}, Negative: {len(negatives)}, Total: {len(full_dataset)}")
    print(f"Saved to {output_path}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, choices=['ollvm', 'tigress'],
                        help="Choose dataset: 'ollvm' or 'tigress'")
    args = parser.parse_args()

    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    output_dir = os.path.join(parent_dir, "dataset")
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, f"train_{args.dataset_name}.json")
    test_path = os.path.join(output_dir, f"test_{args.dataset_name}.json")
    train_data = load_json(train_path)
    test_data = load_json(test_path)

    generate_finetune_dataset("train", args.dataset_name, train_data, output_dir)
    generate_finetune_dataset("test", args.dataset_name, test_data, output_dir)
