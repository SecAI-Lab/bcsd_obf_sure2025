import json
import random
from pathlib import Path
from collections import Counter

random.seed(42)

OLLVM_PATH = "../dataset/finetuning_test_dataset_ollvm.json"
TIGRESS_PATH = "../dataset/finetuning_test_dataset_tigress.json"

BENIGN_OBFUSCATOR = "none"

TASKS = [
    ("OLLVM", "sub", "clang"),
    ("OLLVM", "fla", "clang"),
    ("OLLVM", "bcf", "clang"),
    ("OLLVM", "all", "clang"),
    ("Tigress", "EncodeArithmetic", "gcc"),
    ("Tigress", "AddOpaque", "gcc"),
    ("Tigress", "EncodeBranches", "gcc"),
    ("Tigress", "Flatten", "gcc"),
    ("Tigress", "Virtualize", "gcc"),
]

def load_dataset(dataset_name):
    path = OLLVM_PATH if dataset_name == "OLLVM" else TIGRESS_PATH
    with open(path, "r") as f:
        return json.load(f)

def filter_pairs(data, obf1, obf2, compiler):
    pos_pairs = []
    neg_pairs = []

    for item in data:
        f1, f2 = item["func1"], item["func2"]
        if f1["compiler"] != compiler or f2["compiler"] != compiler:
            continue

        obfs = {f1["obfuscator"], f2["obfuscator"]}
        if obfs == {obf1, obf2}:
            if item["label"] == 1:
                pos_pairs.append(item)
            else:
                neg_pairs.append(item)

    return pos_pairs, neg_pairs

def save_dataset(dataset, name):
    path = f"../dataset/{name}.json"
    with open(path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"[{name}] => Saved {len(dataset)} pairs.")

def process_task(dataset_name, target_obf, compiler, max_samples=1000):
    data = load_dataset(dataset_name)
    pos_pairs, neg_pairs = filter_pairs(data, target_obf, BENIGN_OBFUSCATOR, compiler)

    n = min(max_samples, len(pos_pairs), len(neg_pairs))
    pos_selected = random.sample(pos_pairs, n)
    neg_selected = random.sample(neg_pairs, n)

    final_dataset = pos_selected + neg_selected
    random.shuffle(final_dataset)

    out_name = f"{dataset_name}_{target_obf}_{BENIGN_OBFUSCATOR}_{compiler}"
    save_dataset(final_dataset, out_name)
    print(f"  Positive: {len(pos_selected)}, Negative: {len(neg_selected)}\n")

def process_benign_benign(dataset_name, compiler, max_pos=2500):
    data = load_dataset(dataset_name)
    pos_pairs, neg_pairs = filter_pairs(data, BENIGN_OBFUSCATOR, BENIGN_OBFUSCATOR, compiler)

    n = min(max_pos, len(pos_pairs), len(neg_pairs))
    pos_selected = random.sample(pos_pairs, n)
    neg_selected = random.sample(neg_pairs, n)

    final_dataset = pos_selected + neg_selected
    random.shuffle(final_dataset)

    out_name = f"{dataset_name}_benign_benign_{compiler}"
    save_dataset(final_dataset, out_name)
    print(f"  Positive: {len(pos_selected)}, Negative: {len(neg_selected)}\n")

def process_all_benign_gcc():
    data = load_dataset("Tigress")
    pos_pairs, neg_pairs = filter_pairs(data, BENIGN_OBFUSCATOR, BENIGN_OBFUSCATOR, "gcc")

    n = min(len(pos_pairs), len(neg_pairs))
    if n == 0:
        print(f"[Tigress_benign_benign_gcc_full] => Saved 0 pairs.\n")
        return

    pos_selected = random.sample(pos_pairs, n)
    neg_selected = random.sample(neg_pairs, n)

    final_dataset = pos_selected + neg_selected
    random.shuffle(final_dataset)

    out_name = f"{dataset_name}_benign_benign_{compiler}"
    save_dataset(final_dataset, out_name)
    print(f"  Positive: {len(pos_selected)}, Negative: {len(neg_selected)}\n")

if __name__ == "__main__":
    for dataset_name, obf, compiler in TASKS:
        print(f"Processing: [{dataset_name}_{obf}, {BENIGN_OBFUSCATOR}; {compiler}]")
        process_task(dataset_name, obf, compiler)

    print("Processing: [Benign; clang, Benign; clang]")
    process_benign_benign("OLLVM", "clang", max_pos=2500)

    print("Processing: [Benign; gcc, Benign; gcc]")
    process_all_benign_gcc()
