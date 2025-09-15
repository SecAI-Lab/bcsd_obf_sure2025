import os
import json
import argparse
from collections import defaultdict
from tqdm import tqdm

def process_dataset(dir_path, output_f_name):
    opt_bin_map = defaultdict(lambda: defaultdict(set))
    fn_count_by_obf = defaultdict(int)
    fn_count_by_obf_opt = defaultdict(lambda: defaultdict(int))
    corpus_by_obf = defaultdict(set)

    fn_per_bin = defaultdict(set)
    fn_per_bin_count = defaultdict(int)

    f_li = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            if f.endswith('.json'):
                f_li.append(os.path.join(root, f))

    for file_path in tqdm(f_li, desc=f"Processing JSON files in {dir_path}"):
        f_name = os.path.basename(file_path)
        parts = f_name.split('__')
        if len(parts) != 4:
            continue

        corpus_version = parts[0]
        binary_name = parts[1]
        optimizer = parts[2]
        obfuscator = parts[3].replace('.json', '')
        corpus = corpus_version.split('-')[0]

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                fn_names = list(data.keys())
                fn_count = len(fn_names)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        opt_bin_map[obfuscator][optimizer].add(binary_name)
        corpus_by_obf[obfuscator].add(corpus)
        fn_count_by_obf[obfuscator] += fn_count
        fn_count_by_obf_opt[obfuscator][optimizer] += fn_count

        for fn in fn_names:
            fn_per_bin[binary_name].add(fn)

    for fn_set in fn_per_bin.values():
        for fn in fn_set:
            fn_per_bin_count[fn] += 1

    total_bins = 0
    total_fns = 0
    for obf in opt_bin_map:
        for opt in opt_bin_map[obf]:
            total_bins += len(opt_bin_map[obf][opt])
            total_fns += fn_count_by_obf_opt[obf][opt]

    unique_bin_count = len(fn_per_bin)

    common_fns = {
        fn: count for fn, count in fn_per_bin_count.items()
        if count >= unique_bin_count * 0.9
    }

    with open(output_f_name, 'w') as f:
        f.write("Binary count by obfuscator:\n")
        for obf in sorted(opt_bin_map):
            corpus_li = sorted(corpus_by_obf[obf])

            total_bin_count = sum(len(opt_bin_map[obf][opt]) for opt in opt_bin_map[obf])
            total_func_count = fn_count_by_obf[obf]

            f.write(f"{obf}: {total_bin_count} binaries, {total_func_count} functions\n")
            f.write(f"  corpus: {len(corpus_li)} ({', '.join(corpus_li)})\n")

            for opt in sorted(opt_bin_map[obf]):
                bin_count = len(opt_bin_map[obf][opt])
                func_count = fn_count_by_obf_opt[obf][opt]
                f.write(f"  {opt}: {bin_count} binaries, {func_count} functions\n")

        f.write(f"\nTotal binaries: {total_bins} (unique: {unique_bin_count})\n")
        f.write(f"Total functions: {total_fns}\n")

        f.write("\nCommon functions (appeared in ≥ 90% of unique binaries):\n")
        f.write(f"Total common functions: {len(common_fns)}\n")
        for fn in sorted(common_fns, key=lambda x: -common_fns[x]):
            f.write(f"  {fn}: {common_fns[fn]}/{unique_bin_count}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze function statistics by dataset")
    parser.add_argument('--dataset_name', type=str, required=True, choices=['ollvm', 'tigress'],
                        help="Dataset name to process: 'ollvm' or 'tigress'")
    args = parser.parse_args()

    dataset_name = args.dataset_name
    dataset_path = os.path.join('/data/sure2025', dataset_name)
    output_file = f"data_info_{dataset_name}.txt"

    process_dataset(dataset_path, output_file)
