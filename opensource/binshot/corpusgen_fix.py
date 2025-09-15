################################################################
# ASM + Vocabulary Extractor for Binary Normalization Dataset #
# Modified from: Practical Binary Code Similarity Detection    #
# Author: (original) Sunwoo Ahn, Hyungjoon Koo                #
# Modified by: ChatGPT (Python 3.5 compatible version)        #
################################################################
import os
import sys
import pickle
import argparse
import tqdm
import normalize
import unit
import util
from collections import Counter

def collect_targets(directory):
    targets = []
    for fname in os.listdir(directory):
        full_path = os.path.join(directory, fname)
        if os.path.isfile(full_path):
            skip_ext = ['.dmp.gz', '.id0', '.id1', '.id2', '.nam', '.til', '.pkl', '.i64']
            if not any(fname.endswith(ext) for ext in skip_ext):
                targets.append(full_path)
    return sorted(targets)

def run_normalization(target, pkl_dir, normalization_level=3):
    ida_dmp_path = target + ".dmp.gz"
    if not os.path.exists(ida_dmp_path):
        print("[-] Missing dmp file for: {}".format(target))
        return None
    bin_info = unit.Binary_Info(target)
    bin_info.compiler_info_label = "unknown"
    bin_info.opt_level_label = "unknown"
    print("[+] Normalizing: {}".format(target))
    nn = normalize.Normalization(util.load_from_dmp(ida_dmp_path))
    nn.build_bininfo(bin_info)
    nn.disassemble_and_normalize_instructions(normalization_level=normalization_level)
    BS = nn.pickle_dump(unit.BinarySummary())
    basename = os.path.basename(target)
    pickle.dump(BS, open(os.path.join(pkl_dir, basename + ".pkl"), 'wb'))
    return BS

def generate_corpus_and_voca(targets, pkl_dir, corpus_path, voca_path):
    total_voca = Counter()
    total_lines = []
    for target in tqdm.tqdm(targets, desc="Processing binaries"):
        basename = os.path.basename(target)
        pkl_file = os.path.join(pkl_dir, basename + ".pkl")
        if os.path.exists(pkl_file):
            BS = pickle.load(open(pkl_file, 'rb'))
        else:
            result = run_normalization(target, pkl_dir)
            if result is None:
                continue
            BS = result
        for fs in BS.fns_summaries:
            if fs.is_linker_func:
                continue
            normalized_instrs = [x for x in filter(None, fs.normalized_instrs)]
            line = "{}\t{}\t{}\n".format(basename, fs.fn_name, ','.join(normalized_instrs))
            total_lines.append(line)
            total_voca += Counter(normalized_instrs)
    with open(corpus_path, 'w') as f:
        f.writelines(total_lines)
    with open(voca_path, 'w') as f:
        for instr, count in sorted(total_voca.items()):
            f.write("{},{}\n".format(instr, count))
    print("[+] Total functions processed: {}".format(len(total_lines)))
    print("[+] Vocabulary size: {}".format(len(total_voca)))

def main():
    parser = argparse.ArgumentParser("ASM & VOCA Extractor")
    parser.add_argument("-d", "--binary_dir", type=str, required=True, help="Directory of binaries")
    parser.add_argument("-pkl", "--pkl_dir", type=str, required=True, help="Directory to store/load pickle files")
    parser.add_argument("-o", "--corpus_dir", type=str, required=True, help="Directory to save corpus & voca")
    parser.add_argument("-n", "--norm_level", type=int, default=3, help="Normalization level (default 3)")
    args = parser.parse_args()
    if not os.path.exists(args.pkl_dir):
        os.makedirs(args.pkl_dir)
    if not os.path.exists(args.corpus_dir):
        os.makedirs(args.corpus_dir)
    targets = collect_targets(args.binary_dir)
    dataset_name = os.path.basename(os.path.normpath(args.binary_dir))
    corpus_file = os.path.join(args.corpus_dir, "{}.corpus.txt".format(dataset_name))
    voca_file = os.path.join(args.corpus_dir, "{}.voca.txt".format(dataset_name))
    generate_corpus_and_voca(targets, args.pkl_dir, corpus_file, voca_file)

if __name__ == "__main__":
    main()