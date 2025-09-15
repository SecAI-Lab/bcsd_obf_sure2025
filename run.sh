
#Dataset preprocessing
# output_files: train.json, test.json, train_test_data_info.txt
python make_dataset.py --dataset_name ollvm
python make_dataset.py --dataset_name tigress

python make_tokenizer_dataset.py

python make_pretrain_dataset.py --dataset_name ollvm
# [Done] Saved 710084 asm_code entries to: /home/yujeong/project/sure2025_ver2/dataset/pretrain_train_dataset_ollvm.txt

python make_pretrain_dataset.py --dataset_name tigress
# [Done] Saved 52362 asm_code entries to: /home/yujeong/project/sure2025_ver2/dataset/pretrain_train_dataset_tigress.txt

# create finetuning dataset
# input: train.json, test.json
# output: finetuning_train_dataset.json, finetuning_test_dataset.json
python make_finetune_dataset.py --dataset_name tigress
python make_finetune_dataset.py --dataset_name ollvm

# pretrain
python pretrain.py --dataset_name tigress --device 1 
python pretrain.py --dataset_name ollvm --device 0

# finetuning
python finetune.py --dataset_name tigress --device 1 2>&1 | tee ./finetune_output_tigress.txt
python finetune.py --dataset_name ollvm --device 0 2>&1 | tee ./finetune_output_ollvm.txt

#validation finetuning dataset
#input: finetuning_train_dataset.json, finetuning_test_dataset.json
#output: heatmap
python val_finetunedata.py --dataset_name tigress
python val_finetunedata.py --dataset_name ollvm

# create RQ test datset
python make_rq_test_data.py

# Evaluation using RQ test datset
python eval.py >> rq_eval_out.txt