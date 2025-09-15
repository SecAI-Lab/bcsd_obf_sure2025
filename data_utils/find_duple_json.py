import json

def count_identical_func_pairs(data):
    identical_count = 0

    for entry in data:
        func1 = entry["func1"]
        func2 = entry["func2"]

        keys_to_check = ["binary_name", "function_name", "obfuscator",
                         "corpus", "optimizer_level", "compiler"]

        is_identical = all(func1.get(k) == func2.get(k) for k in keys_to_check)

        if is_identical:
            identical_count += 1

    return identical_count


f_path = "../dataset/finetuning_test_dataset_ollvm.json" 
with open(f_path, "r") as f:
    data = json.load(f)

print(f_path)
count = count_identical_func_pairs(data)
print(f"[✓] Number of samples where func1 and func2 are identical (excluding asm_code): {count}")
