import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding


from .arguments import DataArguments
from .trainer import TevatronTrainer

import logging
logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)
        self.do_distill = self.data_args.training_method != "cross_entropy"

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.prepare_for_model(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry = group['query']
        encoded_query = self.create_one_example(qry, is_query=True)

        encoded_passages = []
        group_positives = group['positives']
        group_negatives = group['negatives']
        all_teacher_scores = []

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
            if self.do_distill:
                pos_teacher_score = group['positive_teacher_scores'][0]
                all_teacher_scores.append(pos_teacher_score)
        else:
            seed = (_hashed_seed + epoch) % len(group_positives)
            pos_psg = group_positives[seed]
            if self.do_distill:
                pos_teacher_score = group['positive_teacher_scores'][seed]
                all_teacher_scores.append(pos_teacher_score)

        encoded_passages.append(self.create_one_example(pos_psg))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
            if self.do_distill:
                assert False
        elif self.data_args.train_n_passages == 1:
            negs = []
            assert False
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
            assert False
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]
            
            if self.do_distill:
                neg_teacher_scores = group['negative_teacher_scores']
                random.Random(_hashed_seed).shuffle(neg_teacher_scores)
                neg_teacher_scores = neg_teacher_scores * 2
                neg_teacher_scores = neg_teacher_scores[_offset: _offset + negative_size]
                all_teacher_scores += neg_teacher_scores

        for neg_psg in negs:
            encoded_passages.append(self.create_one_example(neg_psg))
            
        # print(" ".join(self.tok.convert_ids_to_tokens(encoded_query['input_ids'])))
        # for x in encoded_passages:
        #     print(" ".join(self.tok.convert_ids_to_tokens(x['input_ids'])))
        # print(all_teacher_scores)
        # assert False

        return encoded_query, encoded_passages, all_teacher_scores


class EncodeDataset(Dataset):
    input_keys = ['text_id', 'text']

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_text = self.tok.prepare_for_model(
            text,
            max_length=self.max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
        )
        return text_id, encoded_text


@dataclass
class QPCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    max_q_len: int = 32
    max_p_len: int = 128
    training_method: str = "cross_entropy"

    def __call__(self, features):
        qq = [f[0] for f in features]
        dd = [f[1] for f in features]

        if isinstance(qq[0], list):
            qq = sum(qq, [])
        if isinstance(dd[0], list):
            dd = sum(dd, [])

        q_collated = self.tokenizer.pad(
            qq,
            padding='max_length',
            max_length=self.max_q_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            dd,
            padding='max_length',
            max_length=self.max_p_len,
            return_tensors="pt",
        )

        if self.training_method != "cross_entropy":
            teacher_scores = [f[2] for f in features]
            # TODO: optionally convert to torch tensors here
        else:
            teacher_scores = []

        return q_collated, d_collated, torch.FloatTensor(teacher_scores)


@dataclass
class EncodeCollator(DataCollatorWithPadding):
    def __call__(self, features):
        text_ids = [x[0] for x in features]
        text_features = [x[1] for x in features]
        collated_features = super().__call__(text_features)
        return text_ids, collated_features