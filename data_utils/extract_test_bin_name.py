import json

def extract_bin_name(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    binary_files_to_copy = set()

    for item in data:
        for func_key in ['func1', 'func2']:
            func = item.get(func_key, {})
            corpus = func.get('corpus')
            binary_name = func.get('binary_name')
            optimizer_level = func.get('optimizer_level')
            obfuscator = func.get('obfuscator')

            if None in (corpus, binary_name, optimizer_level, obfuscator):
                continue

            filename = f"{corpus}__{binary_name}__{optimizer_level}__{obfuscator}"
            binary_files_to_copy.add(filename)

    binary_files_list = sorted(binary_files_to_copy)
    return binary_files_list

if __name__ == "__main__":
    tigress_j_path = '../dataset/finetuning_test_dataset_tigress.json'
    ollvm_json_path = '../dataset/finetuning_test_dataset_ollvm.json'

    tigress_bin_name = extract_bin_name(tigress_j_path)
    ollvm_bin_name = extract_bin_name(ollvm_json_path)

    result = {
        "tigress_bin_name": tigress_bin_name,
        "ollvm_bin_name": ollvm_bin_name
    }

    output_path = './test_dataset_bin_file_list.json'
    with open(output_path, 'w') as outfile:
        json.dump(result, outfile, indent=4)

    print(f"Saved bin list: {output_path}")
