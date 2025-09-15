## OBF-BCSD

RoBERTa-based Obfuscated Binary Code Similarity Detection

### Requirements
```
pip install -r requirements.txt
```

Besides, you need to install the followings:
* python3 (tested on 3.8)
* IDA Pro (tested on 8.2)


### Testing
If you only want to test without preprocessing and training:
1. Download the RQ test dataset from [https://zenodo.org/records/17119870](https://zenodo.org/records/17119870).

2. Then move the files into the `dataset` directory:
```
mkdir dataset
mv RQ_test_dataset dataset/
mv dataset/RQ_test_dataset/* dataset/
rmdir dataset/RQ_test_dataset
```
3. Run the evaluation:
```
python eval.py
```



### Dataset Preprocessing
You can download the binary datasets (ollvm.tar.xz and tigress.tar.xz) from 

[https://zenodo.org/records/17119870](https://zenodo.org/records/17119870) and use them to perform the preprocessing steps for training and testing.

By default, put `ollvm` and `tigress` under the `/data` directory.

#### Step 1 — Generate initial dataset
```
python make_dataset.py --dataset_name ollvm
python make_dataset.py --dataset_name tigress
```

#### Step 2 — Generate tokenizer dataset
```
python make_tokenizer_dataset.py
```

#### Step 3 — Generate pretraining dataset
```
python make_pretrain_dataset.py --dataset_name ollvm
python make_pretrain_dataset.py --dataset_name tigress
```

#### Step 4 — Generate finetuning dataset
```
python make_finetune_dataset.py --dataset_name ollvm
python make_finetune_dataset.py --dataset_name tigress
```

---
### Pretraining
```
python pretrain.py --dataset_name tigress
python pretrain.py --dataset_name ollvm
```

### Finetuning
```
python finetune.py --dataset_name tigress 
python finetune.py --dataset_name ollvm
```

### Finetuning Dataset Validation
```
python val_finetunedata.py --dataset_name tigress
python val_finetunedata.py --dataset_name ollvm
```

### Create RQ Test Dataset
```
python make_rq_test_data.py
```

### Evaluation
```
python eval.py
```

