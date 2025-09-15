import os
import json
import csv
from collections import defaultdict

def extract_package_version(corpus):
    if '-' in corpus:
        package, version = corpus.split('-', 1)
    else:
        package, version = corpus, ''
    return package, version

def normalize_obf(obf, source_file):
    if 'tigress' in source_file:
        return 'gcc-none' if obf == 'none' else obf
    else:
        return 'clang-none' if obf == 'none' else obf

def collect_statistics(json_files):
    stat = defaultdict(lambda: {
        'binaries': set(),
        'binary_opt_pairs': set(),  # [MODIFIED] Store (binary_name, opt_level) pairs
        'O0': 0,
        'O1': 0,
        'O2': 0,
        'O3': 0,
        'Total': 0
    })

    valid_opt_levels = ['O0', 'O1', 'O2', 'O3']

    for json_file in json_files:
        print(f"Reading {json_file}")
        with open(json_file, 'r') as f:
            data = json.load(f)

        for item in data:
            corpus = item['corpus']
            package, version = extract_package_version(corpus)
            bin_name = item['binary_name']
            raw_obf = item['obfuscator']
            obf = normalize_obf(raw_obf, json_file)
            opt_level_key = item['optimizer_level']

            key = (package, version, obf)
            stat[key]['binaries'].add(bin_name)
            stat[key]['binary_opt_pairs'].add((bin_name, opt_level_key))  # [MODIFIED] Track each binary+opt combo

            if opt_level_key in valid_opt_levels:
                stat[key][opt_level_key] += 1
            else:
                print(f"[Warning] Unknown optimization level: {opt_level_key}")
            stat[key]['Total'] += 1

    return stat

def save_statistics_to_csv(stat, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        # [MODIFIED] Added 'total_binary' to header
        fieldnames = ['package', 'version', 'binaries', 'total_binary', 'obf', 'O0', 'O1', 'O2', 'O3', 'Total']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        sorted_items = sorted(stat.items(), key=lambda x: (x[0][0], x[0][2]))

        for (package, version, obf), values in sorted_items:
            row = {
                'package': package,
                'version': version,
                'binaries': len(values['binaries']),
                'total_binary': len(values['binary_opt_pairs']),  # [MODIFIED]
                'obf': obf,
                'O0': values['O0'],
                'O1': values['O1'],
                'O2': values['O2'],
                'O3': values['O3'],
                'Total': values['Total']
            }
            writer.writerow(row)

if __name__ == '__main__':
    json_files = [
        '../dataset/train_ollvm.json',
        '../dataset/test_ollvm.json',
        '../dataset/train_tigress.json',
        '../dataset/test_tigress.json'
    ]

    stat = collect_statistics(json_files)
    save_statistics_to_csv(stat, 'dataset_summary_new.csv')


# with open('../dataset/train_ollvm.json') as f:
#     loaded_train = json.load(f)

# print(f"Loaded JSON train count: {len(loaded_train)}")