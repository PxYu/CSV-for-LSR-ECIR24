# CSV-for-LSR-ECIR24

## Introduction

Welcome to the official repository for ECIR '24 paper, [Improved Learned Sparse Retrieval with Corpus-Specific Vocabularies](https://arxiv.org/abs/2401.06703). This contains code and instructions for fully reproducing our results, as well as pointers to checkpoints and datasets that you could potentially incorporate into your workflow!

## Overview

In general, there are 6 steps to fully reproduce our results:

1. Learning corpus-specific vocabularies (CSV).
2. Pre-training CSV-based language models on the retrieval corpus.
3. Training and expanding the corpus using TILDE based on the CSV-based LM.
4. Creating expanded corpus and training data for uniCOIL.
5. Training uniCOIL based on CSV-based LM.
6. Inferencing uniCOIL on the expanded corpus and creating the inverted index.

## Resources

Under a lot of circumstances, you don't actually need the follow the whole workflow described above. For example, you could take one of the CSV-based pre-trained checkpoints and finetune it with SPLADE. Thus, we provide several checkpoints and datasets that you could potential plug into your workflows and try them out!

| Type | Link | Meaning |
| ---- | ---- | ------- |
| Pre-trained model | [pxyu/MSMARCO-V2-BERT-MLM-CSV30k](https://huggingface.co/pxyu/MSMARCO-V2-BERT-MLM-CSV30k) | BERT (CSV, 30K MS MARCO vocabularies) that is pre-trained on MS MARCO v2 corpus for 3 epochs |
| Pre-trained model | [pxyu/MSMARCO-V2-BERT-MLM-CSV100k](https://huggingface.co/pxyu/MSMARCO-V2-BERT-MLM-CSV100k) | BERT (CSV, 100K MS MARCO vocabularies) that is pre-trained on MS MARCO v2 corpus for 3 epochs |
| Pre-trained model | [pxyu/MSMARCO-V1-BERT-MLM-CSV300k](https://huggingface.co/pxyu/MSMARCO-V1-BERT-MLM-CSV300k) | BERT (CSV, 300K MS MARCO vocabularies) that is pre-trained on MS MARCO v1 corpus for 10 epochs |
| Pre-trained model | [pxyu/MSMARCO-V2-BERT-MLM-CSV300k](https://huggingface.co/pxyu/MSMARCO-V2-BERT-MLM-CSV300k) | BERT (CSV, 300K MS MARCO vocabularies) that is pre-trained on MS MARCO v2 corpus for 3 epochs |

(more to be added...)

## Detailed steps of the 7-step approach

### 1. Learning corpus-specific vocabularies (CSV)

### 2. Pre-training CSV-based language models on the retrieval corpus.

### 3. Training and expanding the corpus using TILDE based on the CSV-based LM.

By now, you should have a BERT checkpoint with CSV (or a Huggingface checkpoint we shared) that is pre-trained on the retrieval corpus, located at `PRETRAINED_BERT_PATH_OR_NAME`. For document expansion, we mostly inherit the code from [TILDE framework](https://github.com/ielab/TILDE), based on which we make some changes to generate augmented training data in order to deal with the false-negative issue in MS MARCO.

For training a TILDE model using MS MARCO, we first need to process the data:

```shell
cd tilde
mkdir -p data/hard_neg data/train data/collection

# download MS MARCO data and place them into the data folder
wget https://huggingface.co/datasets/sentence-transformers/msmarco-hard-negatives/resolve/main/msmarco-hard-negatives.jsonl.gz data/hard_neg
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz data/hard_neg
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz data/collection
tar -xzvf data/hard_neg/queries.tar.gz -C data/hard_neg/
tar -xzvf data/collection/collection.tar.gz -C data/collection/

# create augmented training data based on borda aggregatation
cd scripts
python create-borda-falneg.py --top_k 10
```

Now, we have augmented training data for TILDE at `tilde/data/train/borda_top10.train.tsv`. The next step is to train a BERT-based TILDE model using this data:

```shell
# go back the the tilde folder
cd tilde

python train_tilde.py \
    --model_type_or_path PRETRAINED_BERT_PATH_OR_NAME \
    --train_path data/train/borda_top10.train.tsv \
    --save_path checkpoints/YOUR_TILDE_MODEL_NAME \
    --batch_size 64 \
    --num_gpus 8 \
    --use_dl --use_ql
```

Finally, we can use the trained BERT model to acquire the expanded tokens that should be added to every MS MARCO passage:

```shell
python expansion.py \
    --model_type_or_path PRETRAINED_BERT_PATH_OR_NAME \
    --ckpt_path checkpoints/YOUR_TILDE_MODEL_NAME/epoch_5.ckpt \
    --corpus_path data/collection/collection.tsv \
    --output_dir data/collection/expanded/YOUR_TILDE_MODEL_NAME \
    --topk 200 \
    --batch_size 64 \
    --shard -1 \
    --num_workers 8 \
    --store_raw
```

Now, the expanded terms related to our new vocabularies are available at `tilde/data/collection/expanded/YOUR_TILDE_MODEL_NAME`, which is valuable for training effective uniCOIL next.

### 4. Creating expanded corpus and training data for uniCOIL.


### 5. Training uniCOIL based on CSV-based LM.

install uniCOIL in the locally editable way: 

```shell
cd unicoil
pip install -e .
```

Recommended environments:
```
pytorch-lightning==1.6.5
torch==1.13.1
transformers==4.31.0
```

(Optional, you could just use a pre-trained checkpoint listed in step 6) Then train using something like:

```python
python -m torch.distributed.launch --nproc_per_node=4 examples/unicoil/train_unicoil.py \
  --output_dir trained_files/tilde-bert100k-borda10-kldiv \
  --model_name_or_path pxyu/MSMARCO-V2-BERT-MLM-CSV100k \
  --save_steps 10000 \
  --train_dir data/tilde-bert100k-borda10/training/distill-data.jsonl \
  --training_method kl_div \
  --dataset_name json \
  --fp16 \
  --dataset_proc_num 12 \
  --per_device_train_batch_size 8 \
  --train_n_passages 8 \
  --learning_rate 5e-6 \
  --q_max_len 16 \
  --p_max_len 384 \
  --num_train_epochs 5 \
  --add_pooler \
  --projection_in_dim 768 \
  --projection_out_dim 1 \
  --logging_steps 500 \
  --overwrite_output_dir
```

### 6. Inferencing uniCOIL on the expanded corpus and creating the inverted index.

If you ran the last step successfully, you should have `trained_files/tilde-bert100k-borda10-kldiv` available for inference. Alternatively, you could just download a pre-trained checkpoint we shared on Huggingface Hub, like:

```shell 
git lfs install
git clone https://huggingface.co/pxyu/CSV100K-TildeA-KLDiv
```

Now, we are able to run inference with the local weights:

```python
import torch
import random
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from tevatron.arguments import DataArguments
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoTokenizer
from tevatron.data import EncodeDataset, EncodeCollator
from tevatron.modeling import EncoderOutput, UniCoilModel

# configure the model

q_max_len = 16
p_max_len = 384
encode_is_qry = False
text_max_length = q_max_len if encode_is_qry else p_max_len

# hf_model_name = "pxyu/UniCOIL-MSMARCO-KL-Distillation-CSV100k"
local_model_name = "trained_files/tilde-bert100k-borda10-kldiv"

config = AutoConfig.from_pretrained(
    local_model_name,
    num_labels=1,
)

tokenizer = AutoTokenizer.from_pretrained(
    local_model_name,
    use_fast=True,
)

model = UniCoilModel.load(
    model_name_or_path=local_model_name,
    config=config,
)

# THIS IS IMPORTANT!
disabled_token_ids = tokenizer.convert_tokens_to_ids(["[SEP]", "[CLS]", "[MASK]", "[PAD]"])
model.disabled_token_ids = disabled_token_ids


# sample 100 documents as examples

marco_passage = load_dataset("Tevatron/msmarco-passage-corpus")['train']

sampled_id2text = {}
for x in tqdm(marco_passage):
    docid = x['docid']
    text = x['text']
    sampled_id2text[docid] = text
    if len(sampled_id2text) == 100:
        break

pairs = [{"text_id": k,  "text": tokenizer.encode(v, add_special_tokens=False)} for k, v in sampled_id2text.items()]
encode_dataset = EncodeDataset(pairs, tokenizer, text_max_length)

encode_loader = DataLoader(
    encode_dataset,
    batch_size=4,
    collate_fn=EncodeCollator(
        tokenizer,
        max_length=text_max_length,
        padding='max_length'
    ),
    shuffle=False,
    drop_last=False,
)

encoded = []
lookup_indices = []
model.eval()
device = "cpu"

import numpy as np
def process_output(example):
    indices = example.nonzero()[0]
    values = example[indices]
    quantized = (np.ceil(values * 100)).astype(int)
    result = {str(i): int(v) for i, v in zip(indices, quantized)}
    return result

for (batch_ids, batch) in tqdm(encode_loader):
    lookup_indices.extend(batch_ids)
    with torch.no_grad():
        for k, v in batch.items():
            batch[k] = v.to(device)
        if encode_is_qry:
            model_output: EncoderOutput = model(query=batch)
            output = model_output.q_reps.cpu().detach().numpy()
        else:
            model_output: EncoderOutput = model(passage=batch)
            output = model_output.p_reps.cpu().detach().numpy()

    encoded += list(map(process_output, output))

```


