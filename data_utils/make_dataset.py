import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
import random
import matplotlib.pyplot as plt

def save_info_fig(split_log, dataset_name):
    labels = [f"{obf}\n{opt}" for obf, opt, _, _, _ in split_log]
    train_vals = [tr for _, _, _, tr, _ in split_log]
    test_vals = [te for _, _, _, _, te in split_log]
    x = range(len(labels))
    bar_width = 0.4

    plt.figure(figsize=(14, 6))
    plt.bar([i - bar_width / 2 for i in x], train_vals, width=bar_width, label='Train')
    plt.bar([i + bar_width / 2 for i in x], test_vals, width=bar_width, label='Test')

    plt.xticks(ticks=x, labels=labels, rotation=45, ha='right')
    plt.ylabel("Number of Samples")
    plt.title("Sample Counts per (Obfuscator, Optimizer Level) in Train/Test Sets")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"train_test_data_info_{dataset_name}.png")
    plt.close()

def find_common_fn(json_dir, threshold=0.9, obf_filter=None):
    fn_per_bin = defaultdict(set)
    f_li = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    for f_name in f_li:
        parts = f_name.replace('.json', '').split('__')
        if len(parts) != 4:
            continue

        _, _, _, obf = parts
        if obf_filter and obf != obf_filter:
            continue

        path = os.path.join(json_dir, f_name)
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                fn_per_bin[f_name] = set(data.keys())
        except Exception as e:
            print(f"Error loading {f_name}: {e}")

    total_bins = len(fn_per_bin)
    fn_occur = defaultdict(int)
    for fn_set in fn_per_bin.values():
        for fn in fn_set:
            fn_occur[fn] += 1

    common_fns = {
        fn for fn, count in fn_occur.items()
        if count >= total_bins * threshold
    }
    return common_fns, total_bins

def make_data(json_dir, dataset_name, common_fns, out_path, d_out_dir):
    output_data = []
    f_li = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    total_fns_before = 0
    total_fns_after = 0
    del_common = 0
    del_short = 0
    bins = set()
    corpus_set = set()
    opt_obf_bin_count = defaultdict(int)

    for f_name in tqdm(f_li, desc=f"Processing JSON files in {dataset_name}"):
        parts = f_name.replace('.json', '').split('__')
        if len(parts) != 4:
            print(f"Invalid file name format: {f_name}")
            continue

        corpus_ver, bin_name, opt_level, obf = parts
        corpus = corpus_ver.split('-')[0]
        compiler = "gcc" if dataset_name == "tigress" else "clang"
        path = os.path.join(json_dir, f_name)

        bins.add(bin_name)
        corpus_set.add(corpus)
        opt_obf_bin_count[(obf, opt_level)] += 1

        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {f_name}: {e}")
            continue

        for fn_name, asm_code in data.items():
            total_fns_before += 1

            if fn_name in common_fns:
                del_common += 1
                continue
            if len(asm_code) <= 5:
                del_short += 1
                continue

            output_data.append({
                "asm_code": ', '.join(asm_code),
                "binary_name": bin_name,
                "function_name": fn_name,
                "obfuscator": obf,
                "corpus": corpus,
                "optimizer_level": opt_level,
                "compiler": compiler
            })
            total_fns_after += 1

    fn_map_global = defaultdict(list)
    for item in output_data:
        fn_map_global[item["function_name"]].append(item)

    fn_items = list(fn_map_global.items())
    random.shuffle(fn_items)

    split_idx = int(0.8 * len(fn_items))
    global_train_fns = set(fn for fn, _ in fn_items[:split_idx])
    global_test_fns = set(fn for fn, _ in fn_items[split_idx:])

    grouped = defaultdict(list)
    for item in output_data:
        key = (item["obfuscator"], item["optimizer_level"])
        grouped[key].append(item)

    train_data, test_data = [], []
    split_log = []

    for (obf, opt), items in grouped.items():
        tr_items = [item for item in items if item["function_name"] in global_train_fns]
        te_items = [item for item in items if item["function_name"] in global_test_fns]

        train_data.extend(tr_items)
        test_data.extend(te_items)

        split_log.append((obf, opt, len(items), len(tr_items), len(te_items)))


    with open(f'{d_out_dir}/train_{dataset_name}.json', 'w') as f:
        json.dump(train_data, f, indent=2)
    with open(f'{d_out_dir}/test_{dataset_name}.json', 'w') as f:
        json.dump(test_data, f, indent=2)

    train_fn_names = {item["function_name"] for item in train_data}
    test_fn_names = {item["function_name"] for item in test_data}
    overlap_fn_names = train_fn_names & test_fn_names

    print(f"[Check] Unique function names in train: {len(train_fn_names)}")
    print(f"[Check] Unique function names in test : {len(test_fn_names)}")
    print(f"[Check] Overlapping function names     : {len(overlap_fn_names)}")

    if overlap_fn_names:
        print(f"[Warning] Found {len(overlap_fn_names)} overlapping function names between train and test!")
        print("Sample overlaps:", list(overlap_fn_names)[:10])
    else:
        print("[OK] No overlapping function names between train and test.")

    with open(out_path, 'w') as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Total unique corpus: {len(corpus_set)}\n")
        f.write(f"Corpus list: {sorted(corpus_set)}\n\n")
        f.write(f"Total binaries (all JSON files): {len(f_li)}\n")
        f.write(f"Unique binary names: {len(bins)}\n")
        f.write("Binary count per (obfuscator, optimizer_level):\n")
        for (obf, opt), count in sorted(opt_obf_bin_count.items()):
            f.write(f"  {obf} / {opt} → {count}\n")
        f.write("\n")
        f.write(f"Total functions before filtering: {total_fns_before}\n")
        f.write(f"Removed due to common functions: {del_common}\n")
        f.write(f"Removed due to ≤5 instructions: {del_short}\n")
        f.write(f"Final functions after filtering: {total_fns_after}\n")
        f.write(f"Train size: {len(train_data)}\n")
        f.write(f"Test size: {len(test_data)}\n")
        f.write(f"Common function count: {len(common_fns)}\n")
        f.write("\nCommon functions:\n")
        f.write(json.dumps(sorted(common_fns)) + "\n\n")
        f.write("Split per (obfuscator, optimizer_level):\n")
        for obf, opt, total, tr, te in split_log:
            f.write(f"  {obf} / {opt} → total: {total}, train: {tr}, test: {te}\n")
        f.write("\nUnique function names:\n")
        f.write(f"  Train: {len(train_fn_names)}\n")
        f.write(f"  Test : {len(test_fn_names)}\n")
        f.write(f"  Overlap: {len(overlap_fn_names)}\n\n")

        f.write("Train function names:\n")
        f.write(json.dumps(sorted(list(train_fn_names))) + "\n\n")

        f.write("Test function names:\n")
        f.write(json.dumps(sorted(list(test_fn_names))) + "\n\n")

    save_info_fig(split_log, dataset_name)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, choices=['ollvm', 'tigress'],
                        help="Choose dataset: 'ollvm' or 'tigress'")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    json_dir = os.path.join('/data/sure2025', dataset_name)
    info_output_path = f'train_test_data_info_{dataset_name}.txt'
    d_out_dir = "../dataset"
    os.makedirs(d_out_dir, exist_ok=True)

    print(f"[Step 1] Extracting common functions from {dataset_name}...")

    if dataset_name == "tigress":
        # common_fns, bin_count = find_common_fn(json_dir, threshold=0.9, obf_filter="tigressnone")
        common_fns, bin_count = find_common_fn(json_dir, threshold=0.9)
    else:
        common_fns, bin_count = find_common_fn(json_dir, threshold=0.9)

    print(f"→ {len(common_fns)} common functions identified across {bin_count} binaries.")

    print(f"[Step 2] Processing data and writing train/test split...")
    make_data(json_dir, dataset_name, common_fns, info_output_path, d_out_dir)

    print(f"[Done] Files created: train_{dataset_name}.json, test_{dataset_name}.json, train_test_data_info_{dataset_name}.txt, train_test_data_info_{dataset_name}.png")
