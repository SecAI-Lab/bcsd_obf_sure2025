import argparse
import json
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from pathlib import Path

LLVM_OBFUSCATORS = ["none", "bcf", "fla", "sub", "all"]
TIGRESS_OBFUSCATORS = ["none", "AddOpaque", "EncodeArithmetic", "EncodeBranches", "Flatten", "Virtualize"]
OPTIMIZER_LEVELS = ["O0", "O1", "O2", "O3"]

def load_data(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_pair_key(func1, func2):
    key1 = (func1['optimizer_level'], func1['obfuscator'])
    key2 = (func2['optimizer_level'], func2['obfuscator'])
    return tuple(sorted([key1, key2]))

def generate_heatmap(pair_counter, configs, dataset_name, pair_type):
    configs = sorted(configs, key=lambda x: (x[1], x[0]))
    
    idx_x = pd.MultiIndex.from_tuples(configs, names=["Optimizer", "Obfuscator"])
    idx_y = pd.MultiIndex.from_tuples(configs[::-1], names=["Optimizer", "Obfuscator"])

    heatmap_data = pd.DataFrame(0, index=idx_y, columns=idx_x)

    for (conf1, conf2), count in pair_counter.items():
        heatmap_data.loc[conf1, conf2] = count
        heatmap_data.loc[conf2, conf1] = count

    plt.figure(figsize=(14, 12))
    
    ax = sns.heatmap(heatmap_data, cmap="viridis", linewidths=0.5, annot=True, fmt="d", square=True)
    plt.title(f"{pair_type.capitalize()} Pair Heatmap ({dataset_name})")

    xlabels = [f"{obf}-{opt}" for opt, obf in idx_x]
    ylabels = [f"{obf}-{opt}" for opt, obf in idx_y]

    ax.set_xticklabels(xlabels, rotation=90)
    ax.set_yticklabels(ylabels, rotation=0)

    plt.tight_layout()

    output_path = f"pair_heatmap_{dataset_name}_{pair_type}.png"
    plt.savefig(output_path)
    plt.close()

    print(f"Heatmap saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate positive/negative pair heatmaps for BCSD")
    parser.add_argument('--dataset_name', type=str, required=True, choices=['ollvm', 'tigress'],
                        help="Choose dataset: 'ollvm' or 'tigress'")
    args = parser.parse_args()

    json_path = f"../dataset/finetuning_test_dataset_{args.dataset_name}.json"
    if not Path(json_path).exists():
        print(f"Error: File not found -> {json_path}")
        exit(1)

    data = load_data(json_path)

    if args.dataset_name == "ollvm":
        obfs = LLVM_OBFUSCATORS
    else:
        obfs = TIGRESS_OBFUSCATORS

    configs = list(itertools.product(OPTIMIZER_LEVELS, obfs))

    pos_counter = Counter()
    neg_counter = Counter()

    for item in data:
        key = get_pair_key(item["func1"], item["func2"])
        if item["label"] == 1:
            pos_counter[key] += 1
        else:
            neg_counter[key] += 1

    generate_heatmap(pos_counter, configs, args.dataset_name, pair_type="positive")
    generate_heatmap(neg_counter, configs, args.dataset_name, pair_type="negative")
