import json
import random
from pathlib import Path

random.seed(42)

INPUT_PATH  = "../dataset/finetuning_test_dataset_tigress.json"
OUTPUT_DIR  = "../dataset"
OUTPUT_NAME = "Tigress_AddOpaque_none_gcc_new.json"
DEL_FUNCTION = "sub_4027B0"

BENIGN = "none"
TARGET_OBF = "AddOpaque"
COMPILER = "gcc"
MAX_SAMPLES = 1000


with open(INPUT_PATH, "r") as f:
    data = json.load(f)

pos_pairs = []
neg_pairs = []
for item in data:
    f1, f2 = item["func1"], item["func2"]
    if f1["compiler"] != COMPILER or f2["compiler"] != COMPILER:
        continue
    obfs = {f1["obfuscator"], f2["obfuscator"]}
    if obfs != {TARGET_OBF, BENIGN}:
        continue
    # Delete sub_4027B0
    if f1["function_name"] == DEL_FUNCTION or f2["function_name"] == DEL_FUNCTION:
        continue
    (pos_pairs if item["label"] == 1 else neg_pairs).append(item)

def dedupe(pairs):
    seen = set()
    unique = []
    for it in pairs:
        key = (
            it["func1"]["function_name"], it["func1"]["binary_name"],
            it["func1"]["optimizer_level"], it["func1"]["obfuscator"],
            it["func2"]["function_name"], it["func2"]["binary_name"],
            it["func2"]["optimizer_level"], it["func2"]["obfuscator"],
            it["label"]
        )
        if key not in seen:
            seen.add(key)
            unique.append(it)
    return unique

pos_pairs = dedupe(pos_pairs)
neg_pairs = dedupe(neg_pairs)

print(f"After dedupe: positive={len(pos_pairs)}, negative={len(neg_pairs)}")

n_pos = min(MAX_SAMPLES, len(pos_pairs))
n_neg = min(MAX_SAMPLES, len(neg_pairs))

selected = random.sample(pos_pairs, n_pos) + random.sample(neg_pairs, n_neg)
random.shuffle(selected)
print(f"Final samples: positive={n_pos}, negative={n_neg}, total={len(selected)}")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
out_path = Path(OUTPUT_DIR) / OUTPUT_NAME
with open(out_path, "w") as f:
    json.dump(selected, f, indent=2)

print(f"[Done] Saved {len(selected)} pairs to {out_path}")
