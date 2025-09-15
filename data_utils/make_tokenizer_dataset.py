
import os
import json

def load_asm_codes(path):
    with open(path, "r") as f:
        data = json.load(f)
        return [item.get("asm_code", "").strip() for item in data if item.get("asm_code", "").strip()]

def save_tokenizer_dataset(output_dir):
    all_paths = [
        f"{output_dir}/train_tigress.json", f"{output_dir}/train_ollvm.json",
        f"{output_dir}/test_tigress.json", f"{output_dir}/test_ollvm.json"
    ]
    all_asm = set()
    for path in all_paths:
        asm_codes = load_asm_codes(path)
        all_asm.update(asm_codes)

    tokenizer_output = os.path.join(output_dir, "tokenizer_dataset.txt")
    with open(tokenizer_output, "w") as f:
        for asm in sorted(all_asm):
            f.write(asm + "\n")

    print(f"[Done] Saved {len(all_asm)} unique asm_code entries to: {tokenizer_output}")


if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    output_dir = os.path.join(parent_dir, "dataset")
    os.makedirs(output_dir, exist_ok=True)

    save_tokenizer_dataset(output_dir)
