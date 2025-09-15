import json
import os
import argparse

TIGRESS_JSON_FILES = [
    "Tigress_benign_benign_gcc.json",
    "Tigress_EncodeBranches_none_gcc.json",
    "Tigress_Virtualize_none_gcc.json",
    "Tigress_AddOpaque_none_gcc.json",
    "Tigress_EncodeArithmetic_none_gcc.json",
    "Tigress_Flatten_none_gcc.json"
]

OLLVM_JSON_FILES = [
    "OLLVM_all_none_clang.json",
    "OLLVM_benign_benign_clang.json",
    "OLLVM_sub_none_clang.json",
    "OLLVM_bcf_none_clang.json",
    "OLLVM_fla_none_clang.json"
]

def parse_txt_line(line):
    parts = line.strip().split('\t')
    if len(parts) != 3:
        print("Warning: invalid line format -> '{}'".format(line.strip()))
        return None

    try:
        meta, function_name, asm_code = parts
        corpus_full, binary_name, optimizer_level, obfuscator = meta.split('__')
        corpus = corpus_full.split('-')[0] if '-' in corpus_full else corpus_full
        asm_code = asm_code.replace(' ', '')
        return {
            'corpus': corpus,
            'binary_name': binary_name,
            'optimizer_level': optimizer_level,
            'obfuscator': obfuscator,
            'function_name': function_name,
            'asm_code': asm_code
        }
    except Exception as e:
        print("Warning: parse error -> '{}', error: {}".format(line.strip(), str(e)))
        return None

def get_compiler_from_filename(txt_filename):
    if 'tigress' in txt_filename.lower():
        return 'gcc'
    elif 'ollvm' in txt_filename.lower():
        return 'clang'
    else:
        return 'unknown'

def read_input_txt(input_txt_path):
    functions = []
    compiler = get_compiler_from_filename(os.path.basename(input_txt_path))
    with open(input_txt_path, 'r') as f:
        for line in f:
            func = parse_txt_line(line)
            if func is None:
                continue
            func['compiler'] = compiler
            functions.append(func)
    return functions

def build_function_map(functions):
    func_map = dict()
    for func in functions:
        key = (func['function_name'], func['binary_name'], func['optimizer_level'], func['obfuscator'])
        func_map[key] = func
    return func_map

def process_all(input_txt_path, reference_json_folder, output_folder, dataset_name):
    functions = read_input_txt(input_txt_path)
    func_map = build_function_map(functions)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    summary_results = []

    if dataset_name == "tigress":
        json_file_li = TIGRESS_JSON_FILES
    elif dataset_name == "ollvm":
        json_file_li = OLLVM_JSON_FILES
    
    for json_file in json_file_li:
        json_path = os.path.join(reference_json_folder, json_file)
        output_path = os.path.join(output_folder, json_file.replace('.json', '.txt'))

        if not os.path.exists(json_path):
            print("JSON file not found: {}".format(json_path))
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)

        total_pairs = len(data)
        missing_pairs = 0
        written_pairs = 0
        missing_pos = 0
        missing_neg = 0
        written_pos = 0
        written_neg = 0

        with open(output_path, 'w') as out_f:
            for pair in data:
                f1 = pair['func1']
                f2 = pair['func2']
                label = pair['label']

                key1 = (f1['function_name'], f1['binary_name'], f1['optimizer_level'], f1['obfuscator'])
                key2 = (f2['function_name'], f2['binary_name'], f2['optimizer_level'], f2['obfuscator'])

                missing = False
                if key1 not in func_map:
                    print("Warning: func1 not found for pair in {}, key: {}".format(json_file, key1))
                    missing = True
                if key2 not in func_map:
                    print("Warning: func2 not found for pair in {}, key: {}".format(json_file, key2))
                    missing = True

                if missing:
                    missing_pairs += 1
                    if label == 1:
                        missing_pos += 1
                    else:
                        missing_neg += 1
                    continue

                func1 = func_map[key1]
                func2 = func_map[key2]

                id_part = "{}:{}_{}_{}:{}_{}_{}".format(
                    func1['function_name'],
                    func1['compiler'], func1['optimizer_level'], func1['obfuscator'],
                    func2['compiler'], func2['optimizer_level'], func2['obfuscator']
                )
                line = "{}\t{}\t{}\t{}\n".format(
                    func1['asm_code'],
                    func2['asm_code'],
                    id_part,
                    label
                )
                out_f.write(line)
                written_pairs += 1
                if label == 1:
                    written_pos += 1
                else:
                    written_neg += 1

        print("Generated: {}".format(output_path))
        summary_results.append((json_file, total_pairs, missing_pairs, written_pairs, missing_pos, missing_neg, written_pos, written_neg))

    print("\n=== Summary ===")
    for json_file, total_pairs, missing_pairs, written_pairs, missing_pos, missing_neg, written_pos, written_neg in summary_results:
        print("{}: total pairs (json) = {}, missing pairs = {} (pos={}, neg={}), final pairs = {} (pos={}, neg={})".format(
            json_file, total_pairs, missing_pairs, missing_pos, missing_neg, written_pairs, written_pos, written_neg
        ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TXT Pair Generator based on JSON reference")
    parser.add_argument("-i", "--input_txt", type=str, required=True, help="Input txt file path")
    parser.add_argument("-j", "--json_folder", type=str, default="../../dataset", help="Reference JSON folder path")
    args = parser.parse_args()

    input_filename = os.path.basename(args.input_txt).lower()
    
    if "tigress" in input_filename:
        subfolder = "tigress"
    elif "ollvm" in input_filename:
        subfolder = "ollvm"
    else:
        subfolder = "unknown"

    out_folder = os.path.join("corpus", subfolder)

    process_all(args.input_txt, args.json_folder, out_folder, subfolder)
