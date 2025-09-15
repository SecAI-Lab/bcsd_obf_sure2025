import os
import json
import argparse


def load_asm_codes(path):
    with open(path, "r") as f:
        data = json.load(f)
        return [item.get("asm_code", "").strip() for item in data if item.get("asm_code", "").strip()]

def save_pretraining_datasets(dataset_name, output_dir):
    train_path = f"{output_dir}/train_{dataset_name}.json"
    pretrain_output = os.path.join(output_dir, f"pretrain_train_dataset_{dataset_name}.txt")

    train_asm = load_asm_codes(train_path)
    with open(pretrain_output, "w") as f:
        for asm in train_asm:
            f.write(asm + "\n")

    print(f"[Done] Saved {len(train_asm)} asm_code entries to: {pretrain_output}")


if __name__ == "__main__":
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
    output_dir = os.path.join(parent_dir, "dataset")
    os.makedirs(output_dir, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, required=True, choices=['ollvm', 'tigress'],
                        help="Choose dataset: 'ollvm' or 'tigress'")
    args = parser.parse_args()

    save_pretraining_datasets(args.dataset_name, output_dir)
