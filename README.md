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
| Pre-trained model | [pxyu/MSMARCO-V2-BERT-MLM-CSV100k](https://huggingface.co/pxyu/MSMARCO-V2-BERT-MLM-CSV30k) | BERT (CSV, 100K MS MARCO vocabularies) that is pre-trained on MS MARCO v2 corpus for 3 epochs |
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
### 6. Inferencing uniCOIL on the expanded corpus and creating the inverted index.



