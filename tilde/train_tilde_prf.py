import pickle
import pytorch_lightning as pl
from transformers import BertTokenizer, AutoTokenizer
import torch
from tools import get_stop_ids
import random
from pytorch_lightning import Trainer, seed_everything
# from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from argparse import ArgumentParser
from torch.utils.data import Dataset, DataLoader
from modeling import TILDE
import os

class CheckpointEveryEpoch(pl.Callback):
    """
    Save a checkpoint every N steps, instead of Lightning's default that checkpoints
    based on validation loss.
    """

    def __init__(
        self,
        start_epoc,
        save_path,
    ):

        self.start_epoc = start_epoc
        self.file_path = save_path

    def on_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train epoch """
        epoch = trainer.current_epoch
        if epoch >= self.start_epoc:
            ckpt_path = os.path.join(self.file_path, f"epoch_{epoch+1}.ckpt")
            trainer.save_checkpoint(ckpt_path)

class MsmarcoDocumentQueryPair(Dataset):
    def __init__(self, path, tokenizer, prf_path, query2id_path, safe_prf):
        self.tokenizer = tokenizer
        self.path = path
        self.queries = []
        self.passages = []
        stop, garbage = get_stop_ids(self.tokenizer)
        self.stop_ids = list(stop)
        self.garbage_ids = garbage
        self.safe_prf = safe_prf

        # print(len(stop), len(garbage))

        with open(path, 'r') as f:
            contents = f.readlines()

        for line in contents:
            passage, query = line.strip().split('\t')

            self.queries.append(query)
            self.passages.append(passage)
            
        with open(prf_path, 'rb') as fin:
            self.prf = pickle.load(fin)

        with open(query2id_path, "rb") as fin:
            self.query2id = pickle.load(fin)

    def __getitem__(self, index):
        query = self.queries[index]
        passage = self.passages[index]

        ## QL

        ind = self.tokenizer(query, add_special_tokens=False)['input_ids']
        feedfack_tokens = list(self.prf[self.query2id[query]])
        feedback_ids = self.tokenizer.convert_tokens_to_ids(feedfack_tokens)
        yq = torch.zeros(self.tokenizer.vocab_size, dtype=torch.float32)
        
        cleaned_ids = []
        yq[self.stop_ids] = -1

        if self.safe_prf:
            # feedback are treated as stop_ids (-1, thus no penalization)
            yq[feedback_ids] = -1
        else:
            # feedback are treated as positive
            ind = ind + feedback_ids

        for id in ind:
            if id not in self.stop_ids and id not in self.garbage_ids:
                cleaned_ids.append(id)
        
        yq[cleaned_ids] = 1
        
        ## DL

        ind = self.tokenizer(passage, add_special_tokens=False)['input_ids']
        cleaned_ids = []
        for id in ind:
            if id not in self.stop_ids and id not in self.garbage_ids:
                cleaned_ids.append(id)
        yd = torch.zeros(self.tokenizer.vocab_size, dtype=torch.float32)
        yd[cleaned_ids] = 1
        yd[self.stop_ids] = -1

        return passage, yq, query, yd

    def __len__(self):
        return len(self.queries)

class MyCollator(object):
    
    def __init__(self, tokenizer, substitute_cls):
        
        self.tokenizer = tokenizer
        self.substitute_cls = substitute_cls
    
    def __call__(self, batch):
        passages = []
        queries = []
        yqs = []
        yds = []

        for passage, yq, query, yd in batch:
            passages.append(passage)
            yqs.append(yq)
            queries.append(query)
            yds.append(yd)

        passage_inputs = self.tokenizer(passages, return_tensors="pt", padding=True, truncation=True, max_length=128)
        passage_input_ids = passage_inputs["input_ids"]
        passage_token_type_ids = passage_inputs["token_type_ids"]
        passage_attention_mask = passage_inputs["attention_mask"]

        query_inputs = self.tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=16)
        query_input_ids = query_inputs["input_ids"]
        query_token_type_ids = query_inputs["token_type_ids"]
        query_attention_mask = query_inputs["attention_mask"]

        if self.substitute_cls:
            passage_input_ids[:, 0] = 1  # 1 is token id for [DOC]
            query_input_ids[:, 0] = 2   # 2 is token id for [QRY]
        
        return passage_input_ids, passage_token_type_ids, passage_attention_mask, torch.stack(yqs), None, \
               query_input_ids, query_token_type_ids, query_attention_mask, torch.stack(yds), None


def main(args):
    
    seed_everything(611)
    
    wandb_logger = WandbLogger(project="DOCEXP-DISTILL")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_type_or_path)

    model = TILDE(
        args.model_type_or_path,
        gradient_checkpointing=args.gradient_checkpoint,
        use_dl=args.use_dl,
        use_ql=args.use_ql
        )
    
    dataset = MsmarcoDocumentQueryPair(args.train_path, tokenizer, args.prf_path, args.query2id_path, args.safe_prf)
    
    my_collator = MyCollator(tokenizer, args.substitute_cls)
    
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        drop_last=True,
                        pin_memory=True,
                        shuffle=True,
                        num_workers=10,
                        collate_fn=my_collator
                        )

    trainer = Trainer(max_epochs=5,
                      devices=args.num_gpus,
                      strategy="ddp",
                    #   precision=16,
                    #   amp_backend="native",
                      checkpoint_callback=False,
                      logger=wandb_logger,
                      accelerator="gpu",
                      callbacks=[CheckpointEveryEpoch(0, args.save_path)]
                      )
    
    trainer.fit(model, loader)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument("--model_type_or_path", default="bert-base-uncased", type=str)
    parser.add_argument("--train_path", required=True)
    parser.add_argument("--prf_path", required=True)
    parser.add_argument("--safe_prf", action="store_true", help="save_prf puts feedback tokens into stopwords level (don't penalize)")
    parser.add_argument("--query2id_path", default="data/prf/query2id.pkl")
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--num_gpus", default=8, type=int)
    parser.add_argument("--use_dl", action='store_true')
    parser.add_argument("--use_ql", action='store_true')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--substitute_cls", action='store_true')
    parser.add_argument("--gradient_checkpoint", action='store_true', help='Ture for trade off training speed for larger batch size')
    args = parser.parse_args()

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    main(args)
