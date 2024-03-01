import json, gzip
from tqdm.auto import tqdm
from datasets import load_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--top_k", type=int, default=10)
args = parser.parse_args()

marco_passage = load_dataset("Tevatron/msmarco-passage-corpus")['train']

id2text = {}

for x in tqdm(marco_passage):
    id2text[x['docid']] = x['text']

meta = {}

with gzip.open("../data/hard_neg/msmarco-hard-negatives.jsonl.gz", 'rt') as fin:
    for line in tqdm(fin):
        tmp = json.loads(line)
        if len(tmp['pos']) < 1:
            continue
        
        vote = {}
        for model, lst in tmp['neg'].items():
            if model == "bm25":
                continue
            cnt = 50
            for docid in lst:
                if docid not in vote:
                    vote[docid] = 0
                vote[docid] += cnt
                cnt -= 1
        
        meta[tmp['qid']] = {
            "pos": tmp['pos'],
            "neg": vote
        }

id2query = {}

with open("../data/hard_neg/queries.train.tsv", 'r') as fin:
    for line in tqdm(fin.readlines()):
        qid, qtext = line.strip().split("\t")
        id2query[qid] = qtext

def write_borda_top_k(k):
    with open(f"../data/train/borda_top{k}.train.tsv", 'w') as fout:
        for qid, d in tqdm(meta.items()):

            qtext = id2query[str(qid)]

            for docid in d['pos']:
                fout.write(f"{id2text[str(docid)]}\t{qtext}\n")
                
            for docid in sorted(d['neg'], key=d['neg'].get, reverse=True)[:k]:
                if docid not in d['pos']:
                    fout.write(f"{id2text[str(docid)]}\t{qtext}\n")

write_borda_top_k(args.top_k)