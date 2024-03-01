from argparse import ArgumentParser
from transformers import BertLMHeadModel, DataCollatorWithPadding, AutoTokenizer
import torch
import json
import re
from nltk.corpus import stopwords
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from modeling import TILDE

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE="cpu"


def clean_vacab(tokenizer, do_stopwords=True):
    if do_stopwords:
        stop_words = set(stopwords.words('english'))
        # keep some common words in ms marco questions
        # stop_words.difference_update(["where", "how", "what", "when", "which", "why", "who"])
        stop_words.add("definition")

    vocab = tokenizer.get_vocab()
    tokens = vocab.keys()

    good_ids = []
    bad_ids = []

    for stop_word in stop_words:
        ids = tokenizer(stop_word, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            bad_ids.append(ids[0])

    for token in tokens:
        token_id = vocab[token]
        if token_id in bad_ids:
            continue

        if len(token) == 1:
            bad_ids.append(token_id)

        if token[0] == '#' and len(token) > 1:
            # good_ids.append(token_id)
            bad_ids.append(token_id)
        else:
            if not re.match("^[A-Za-z0-9_-]*$", token):
                bad_ids.append(token_id)
            else:
                good_ids.append(token_id)
                
    if "##s" in tokenizer.vocab:
        bad_ids.append(tokenizer.convert_tokens_to_ids(["##s"])[0])
    # bad_ids.append(2015)  # add ##s to stopwords
    return good_ids, bad_ids


class MarcoEncodeDataset(Dataset):
    def __init__(self, path, tokenizer, p_max_len=128):
        self.tok = tokenizer
        self.p_max_len = p_max_len
        self.passages = []
        self.pids = []

        with open(path, 'rt') as fin:
            lines = fin.readlines()
            for line in tqdm(lines, desc="Loading collection"):
                pid, passage = line.split("\t")
                self.passages.append(passage)
                self.pids.append(pid)

    def __len__(self):
        return len(self.passages)

    def __getitem__(self, item):
        psg = self.passages[item]
        encoded_psg = self.tok.encode_plus(
            psg,
            max_length=self.p_max_len,
            truncation='only_first',
            return_attention_mask=False,
        )
        # encoded_psg.input_ids[0] = 1  # TILDE use token id 1 as the indicator of passage input.
        return encoded_psg

    def get_pids(self):
        return self.pids


def main(args):
    # model = BertLMHeadModel.from_pretrained("ielab/TILDE", cache_dir='./cache')
    # model = TILDE.load_from_checkpoint(model_type='bert-base-uncased', checkpoint_path=args.ckpt_path).bert
    model = TILDE.load_from_checkpoint(
        model_type=args.model_type_or_path,
        checkpoint_path=args.ckpt_path
    ).bert
    tokenizer = AutoTokenizer.from_pretrained(args.model_type_or_path, use_fast=True, cache_dir='./cache')
    model.eval().to(DEVICE)
    rep, garb = [], []

    file_name = f"expanded-{args.shard}.jsonl" if args.shard != -1 else "expanded.jsonl"

    # with open(os.path.join(args.output_dir, f"{args.shard}.jsonl"), 'w+') as wf:
    with open(os.path.join(args.output_dir, file_name), 'w+') as wf:
        _, bad_ids = clean_vacab(tokenizer)

        encode_dataset = MarcoEncodeDataset(args.corpus_path, tokenizer)
        encode_loader = DataLoader(
            encode_dataset,
            batch_size=args.batch_size,
            collate_fn=DataCollatorWithPadding(
                tokenizer,
                max_length=128,
                padding='max_length'
            ),
            shuffle=False,  # important
            drop_last=False,  # important
            num_workers=args.num_workers,
        )

        pids = encode_dataset.get_pids()
        COUNTER = 0
        for batch in tqdm(encode_loader):
            passage_input_ids = batch.input_ids.numpy()
            batch.to(DEVICE)
            with torch.no_grad():
                logits = model(**batch, return_dict=True).logits[:, 0]
                batch_selected = torch.topk(logits, args.topk).indices.cpu().numpy()

            expansions = []
            for i, selected in enumerate(batch_selected):
                
                non_repeat_terms = np.setdiff1d(selected, passage_input_ids[i], assume_unique=True)
                # print(non_repeat_terms.shape)
                num_repeat = args.topk - non_repeat_terms.shape[0]
                kept_terms = np.setdiff1d(non_repeat_terms, bad_ids, assume_unique=True)
                num_garbage = non_repeat_terms.shape[0] - kept_terms.shape[0]
                # print(kept_terms.shape)
                # print(num_repeat, num_garbage)
                rep.append(num_repeat)
                garb.append(num_garbage)
                
                # expand_term_ids = np.setdiff1d(np.setdiff1d(selected, passage_input_ids[i], assume_unique=True),
                #                                bad_ids, assume_unique=True)
                # print(expand_term_ids.shape)
                # print("top200:", len(selected.tolist()))
                # print("passage:", len(set(passage_input_ids[i].tolist())))
                # print("garbage:", len(bad_ids))
                # print("kept:", expand_term_ids.shape)
                # assert False
                expansions.append(kept_terms)

            for expanded_terms in expansions:
                
                if args.store_raw:
                    
                    expanded_terms = tokenizer.convert_ids_to_tokens(expanded_terms)

                    temp = {
                        "pid": pids[COUNTER],
                        "psg": expanded_terms
                    }

                COUNTER += 1
                wf.write(f'{json.dumps(temp)}\n')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_type_or_path", default="bert-base-uncased", type=str)
    parser.add_argument('--ckpt_path', required=True)
    parser.add_argument('--corpus_path', required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument('--topk', default=200, type=int, help='k tokens with highest likelihood to be expanded to the original document. '
                                                              'NOTE: this is the number before filtering out expanded tokens that already in the original document')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--store_raw', action='store_true', help="True if you want to store expanded raw text. False if you want to expanded store token ids.")
    parser.add_argument('--shard', default=-1, type=int)
    args = parser.parse_args()

    print(args)

    args.output_dir = os.path.join(args.output_dir, f"top{args.topk}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
