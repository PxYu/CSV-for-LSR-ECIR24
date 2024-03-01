import torch
import numpy as np
from transformers import PreTrainedModel, BertConfig, BertModel, BertLMHeadModel, BertTokenizer, AutoTokenizer
from transformers.trainer import Trainer
from torch.utils.data import DataLoader
from typing import Optional
import os
import time

import pytorch_lightning as pl
from tools import get_stop_ids


class TILDE(pl.LightningModule):
    def __init__(self, model_type, from_pretrained=None, gradient_checkpointing=False, use_dl=True, use_ql=True):
        super().__init__()

        self.save_hyperparameters()

        self.use_dl = use_dl  # since we only use Tilde for doc expansion, seems unnecessary to do dl (encode query and match document)
        self.use_ql = use_ql
        if from_pretrained is not None:
            self.bert = BertLMHeadModel.from_pretrained(from_pretrained, cache_dir="./cache")
        else:
            self.config = BertConfig.from_pretrained(model_type, cache_dir="./cache")
            self.config.gradient_checkpointing = gradient_checkpointing  # for trade off training speed for larger batch size
            # self.config.is_decoder = True
            self.bert = BertLMHeadModel.from_pretrained(model_type, config=self.config, cache_dir="./cache")

        self.tokenizer = AutoTokenizer.from_pretrained(model_type, cache_dir="./cache")
        self.num_valid_tok = self.tokenizer.vocab_size - len(get_stop_ids(self.tokenizer))

    def forward(self, x):
        input_ids, token_type_ids, attention_mask = x
        outputs = self.bert(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask,
                            return_dict=True).logits[:, 0]
        return outputs

    # Bi-direction loss (BDQLM)
    def training_step(self, batch, batch_idx):
        # todo test all_gather: self.all_gather()
        passage_input_ids, passage_token_type_ids, passage_attention_mask, yqs, _, \
        query_input_ids, query_token_type_ids, query_ttention_mask, yds, _ = batch

        batch_size = len(yqs)

        if self.use_ql:

            passage_outputs = self.bert(input_ids=passage_input_ids,
                                        token_type_ids=passage_token_type_ids,
                                        attention_mask=passage_attention_mask,
                                        return_dict=True).logits[:, 0]

            batch_passage_prob = torch.sigmoid(passage_outputs)
            passage_pos_loss = 0
            ql_neg_top_prob = []

        if self.use_dl:

            query_outputs = self.bert(input_ids=query_input_ids,
                                    token_type_ids=query_token_type_ids,
                                    attention_mask=query_ttention_mask,
                                    return_dict=True).logits[:, 0]

            batch_query_prob = torch.sigmoid(query_outputs)
            query_pos_loss = 0
            dl_neg_top_prob = []

        for i in range(batch_size):

            if self.use_ql:

                # BCEWithLogitsLoss
                passage_pos_ids_plus = torch.where(yqs[i] == 1)[0]
                passage_pos_ids_minus = torch.where(yqs[i] == 0)[0]

                passage_pos_probs = batch_passage_prob[i][passage_pos_ids_plus]
                passage_neg_probs = batch_passage_prob[i][passage_pos_ids_minus]
                
                # tokenized_neg = self.tokenizer.convert_ids_to_tokens(passage_pos_ids_minus.cpu().tolist())
                # neg_token_to_prob = {k: v for k, v in zip(tokenized_neg, passage_neg_probs.detach().cpu().tolist())}
                # passage_tokens = self.tokenizer.convert_ids_to_tokens(passage_input_ids[0].cpu().tolist())
                # print(passage_tokens)
                # for w in sorted(neg_token_to_prob, key=neg_token_to_prob.get, reverse=True)[:10]:
                #     print(w, f"{neg_token_to_prob[w]:.2f}", w in passage_tokens)
                # # assert False
                
                passage_pos_loss -= torch.sum(torch.log(passage_pos_probs+1e-7)) + torch.sum(torch.log(1-passage_neg_probs+1e-7))
                ql_neg_top_prob.append(torch.mean(torch.topk(passage_neg_probs, 10)[0]).item())

            if self.use_dl:

                query_pos_ids_plus = torch.where(yds[i] == 1)[0]
                query_pos_ids_minus = torch.where(yds[i] == 0)[0]

                query_pos_probs = batch_query_prob[i][query_pos_ids_plus]
                query_neg_probs = batch_query_prob[i][query_pos_ids_minus]
                query_pos_loss -= torch.sum(torch.log(query_pos_probs+1e-7)) + torch.sum(torch.log(1-query_neg_probs+1e-7))
                dl_neg_top_prob.append(torch.mean(torch.topk(query_neg_probs, 10)[0]).item())
        

        if self.use_ql and self.use_dl:
            # print(
            #     f"{passage_pos_loss.item():.1f}", 
            #     f"{query_pos_loss.item():.1f}", 
            #     f"{torch.mean(torch.topk(passage_neg_probs, 10)[0]):.2f}",
            #     f"{torch.mean(torch.topk(query_neg_probs, 10)[0]):.2f}"
            # )
            
            
            # # always the last example from the batch
            # print(
            #     # f"{passage_pos_loss.item():.1f}", 
            #     # f"{query_pos_loss.item():.1f}",
            #     f"{torch.min(passage_pos_probs):.2f}",
            #     f"{torch.min(query_pos_probs):.2f}",
            #     f"{torch.mean(torch.topk(passage_neg_probs, 10)[0]):.2f}",
            #     f"{torch.mean(torch.topk(query_neg_probs, 10)[0]):.2f}"
            # )
            
            
            
            # if torch.isinf(passage_pos_loss):
            #     print(passage_pos_probs, torch.sum(torch.log(passage_pos_probs)).item())
            #     print(passage_neg_probs, torch.sum(torch.log(passage_neg_probs)).item())
            #     assert False
            
            loss = (passage_pos_loss + query_pos_loss) / (self.num_valid_tok * 2)
            
            self.log("ql_neg_top_prob", np.mean(ql_neg_top_prob))
            self.log("dl_neg_top_prob", np.mean(dl_neg_top_prob))
        
        elif self.use_dl and not self.use_ql:
            loss = query_pos_loss / self.num_valid_tok
        elif self.use_ql and not self.use_dl:
            loss = passage_pos_loss / self.num_valid_tok
        else:
            assert False

        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-5)
        return optimizer

    def save(self, path):
        self.bert.save_pretrained(path)
