## Testing BCSD

This is the official repository for BinShot (ACSAC 22') with additional test code for BCSD evaluation.

### Requirements

```
pip install -r requirements.txt
```

Besides, you need to install the followings:
* python3 (tested on 3.8)
* IDA Pro (tested on 7.7)

---
### Tigress Experiment

#### Dataset Preprocessing

```
bash gen_ida.sh ./binary/tigress
bash gen_norm.sh binary/tigress norm/tigress
python3 corpusgen_fix.py -d binary/tigress -pkl norm/tigress -o corpus
python3 voca.py corpus/tigress.voca.txt
python make_finetune_dataset.py --input_txt corpus/tigress.corpus.txt >> out_tigress.txt
```

#### Evaluation

```
ython3 binshot.py -bm models/downstream_full/model_sim/bert_ep19.model -fm models/downstream_full/model_sim/sim_ep19.model -vp corpus/pretrain.all.corpus.voca -r results/tigress -tt corpus/tigress/Tigress_AddOpaque_none_gcc.txt -op models/downstream_full >> results_txt/tigress/Tigress_AddOpaque_none_gcc_result.txt

python3 binshot.py -bm models/downstream_full/model_sim/bert_ep19.model -fm models/downstream_full/model_sim/sim_ep19.model -vp corpus/pretrain.all.corpus.voca -r results/tigress -tt corpus/tigress/Tigress_benign_benign_gcc.txt -op models/downstream_full >> results_txt/tigress/Tigress_benign_benign_gcc_result.txt

python3 binshot.py -bm models/downstream_full/model_sim/bert_ep19.model -fm models/downstream_full/model_sim/sim_ep19.model -vp corpus/pretrain.all.corpus.voca -r results/tigress -tt corpus/tigress/Tigress_EncodeArithmetic_none_gcc.txt -op models/downstream_full >> results_txt/tigress/Tigress_EncodeArithmetic_none_gcc_result.txt

python3 binshot.py -bm models/downstream_full/model_sim/bert_ep19.model -fm models/downstream_full/model_sim/sim_ep19.model -vp corpus/pretrain.all.corpus.voca -r results/tigress -tt corpus/tigress/Tigress_EncodeBranches_none_gcc.txt -op models/downstream_full >> results_txt/tigress/Tigress_EncodeBranches_none_gcc_result.txt

python3 binshot.py -bm models/downstream_full/model_sim/bert_ep19.model -fm models/downstream_full/model_sim/sim_ep19.model -vp corpus/pretrain.all.corpus.voca -r results/tigress -tt corpus/tigress/Tigress_Flatten_none_gcc.txt -op models/downstream_full >> results_txt/tigress/Tigress_Flatten_none_gcc_result.txt

python3 binshot.py -bm models/downstream_full/model_sim/bert_ep19.model -fm models/downstream_full/model_sim/sim_ep19.model -vp corpus/pretrain.all.corpus.voca -r results/tigress -tt corpus/tigress/Tigress_Virtualize_none_gcc.txt -op models/downstream_full >> results_txt/tigress/Tigress_Virtualize_none_gcc_result.txt
```

---

### OLLVM Experiment

#### Dataset Preprocessing
```
bash gen_ida.sh ./binary/ollvm
bash gen_norm.sh binary/ollvm norm/ollvm
python3 corpusgen_fix.py -d binary/ollvm -pkl norm/ollvm -o corpus
python3 voca.py corpus/ollvm.voca.txt
python make_finetune_dataset.py --input_txt corpus/ollvm.corpus.txt >> out_ollvm.txt
```

#### Evaluation
```
python3 binshot.py -bm models/downstream_full/model_sim/bert_ep19.model -fm models/downstream_full/model_sim/sim_ep19.model -vp corpus/pretrain.all.corpus.voca -r results/ollvm -tt corpus/ollvm/OLLVM_all_none_clang.txt -op models/downstream_full >> results_txt/ollvm/OLLVM_all_none_clang_result.txt

python3 binshot.py -bm models/downstream_full/model_sim/bert_ep19.model -fm models/downstream_full/model_sim/sim_ep19.model -vp corpus/pretrain.all.corpus.voca -r results/ollvm -tt corpus/ollvm/OLLVM_bcf_none_clang.txt -op models/downstream_full >> results_txt/ollvm/OLLVM_bcf_none_clang_result.txt

python3 binshot.py -bm models/downstream_full/model_sim/bert_ep19.model -fm models/downstream_full/model_sim/sim_ep19.model -vp corpus/pretrain.all.corpus.voca -r results/ollvm -tt corpus/ollvm/OLLVM_benign_benign_clang.txt -op models/downstream_full >> results_txt/ollvm/OLLVM_benign_benign_clang_result.txt

python3 binshot.py -bm models/downstream_full/model_sim/bert_ep19.model -fm models/downstream_full/model_sim/sim_ep19.model -vp corpus/pretrain.all.corpus.voca -r results/ollvm -tt corpus/ollvm/OLLVM_fla_none_clang.txt -op models/downstream_full >> results_txt/ollvm/OLLVM_fla_none_clang_result.txt

python3 binshot.py -bm models/downstream_full/model_sim/bert_ep19.model -fm models/downstream_full/model_sim/sim_ep19.model -vp corpus/pretrain.all.corpus.voca -r results/ollvm -tt corpus/ollvm/OLLVM_sub_none_clang.txt -op models/downstream_full >> results_txt/ollvm/OLLVM_sub_none_clang_result.txt
```


