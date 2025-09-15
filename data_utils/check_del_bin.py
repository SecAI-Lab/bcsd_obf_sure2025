import os
import json
import argparse
from tqdm import tqdm
from collections import defaultdict

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

def make_data(json_dir, dataset_name, common_fns):
    f_li = [f for f in os.listdir(json_dir) if f.endswith('.json')]

    skipped_bins = []
    obf_opt_per_bin = defaultdict(set)  # { (corpus, bin): set of (obf, opt) present }
    filtered_obf_opt_per_bin = defaultdict(set)  # { (corpus, bin): set of (obf, opt) filtered }

    bin_to_obf_opt = dict()

    for f_name in tqdm(f_li, desc=f"Processing JSON files in {dataset_name}"):
        parts = f_name.replace('.json', '').split('__')
        if len(parts) != 4:
            continue

        corpus_ver, bin_name, opt_level, obf = parts
        corpus = corpus_ver.split('-')[0]
        key = (corpus, bin_name)
        path = os.path.join(json_dir, f_name)

        bin_to_obf_opt[f_name] = (obf, opt_level)
        obf_opt_per_bin[key].add((obf, opt_level))

        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading {f_name}: {e}")
            continue

        kept_fn_count = 0
        for fn_name, asm_code in data.items():
            if fn_name in common_fns:
                continue
            if len(asm_code) <= 5:
                continue
            kept_fn_count += 1

        if kept_fn_count == 0:
            skipped_bins.append(f_name)
            filtered_obf_opt_per_bin[key].add((obf, opt_level))

    fully_removed_obf_count = defaultdict(int)
    for key in obf_opt_per_bin.keys():
        original_set = obf_opt_per_bin[key]
        filtered_set = filtered_obf_opt_per_bin.get(key, set())

        obf_to_opts = defaultdict(set)
        for obf, opt in original_set:
            obf_to_opts[obf].add(opt)

        filtered_obf_to_opts = defaultdict(set)
        for obf, opt in filtered_set:
            filtered_obf_to_opts[obf].add(opt)

        for obf in obf_to_opts:
            expected_opts = {'O0', 'O1', 'O2', 'O3'}
            present_opts = obf_to_opts[obf]
            filtered_opts = filtered_obf_to_opts.get(obf, set())

            if expected_opts.issubset(present_opts) and filtered_opts.issuperset(expected_opts):
                fully_removed_obf_count[obf] += 1

    print("\n[Completely Checked Removed Binaries by Obfuscator (O0~O3 all removed)]")
    for obf, count in sorted(fully_removed_obf_count.items()):
        print(f"  {obf} → {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, choices=['ollvm', 'tigress'],
                        help="Choose dataset: 'ollvm' or 'tigress'")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    json_dir = os.path.join('/data/sure2025', dataset_name)

    print(f"[Step 1] Extracting common functions from {dataset_name}...")
    common_fns, bin_count = find_common_fn(json_dir, threshold=0.9)
    print(f"→ {len(common_fns)} common functions identified across {bin_count} binaries.")

    print(f"[Step 2] Processing data and analyzing full removals...")
    make_data(json_dir, dataset_name, common_fns)

    print(f"[Done] Analysis complete.")
