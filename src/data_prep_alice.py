from datasets import load_dataset 
from typing import Literal
from datasets import load_dataset, enable_caching, Dataset,disable_caching
import json
import pandas as pd
import os 
import torch
from transformers import AutoTokenizer
path_train = "alice_data/ALICE_train_new.csv"
path_ua = "alice_data/ALICE_UA_new.csv"
path_uq = "alice_data/ALICE_UQ_new.csv"
def encoding(example, tokenizer):
    # basic encoding 
    output = tokenizer(
        example["answer"]
    )
    for field in output:
        example[field] = output[field]
    return example
def encoding_with_solution_pair(example, tokenizer):
    # for sequence pair classification
    output = tokenizer(
        example["sample_solution"],
        example["answer"],) 
    for field in output:
        example[field] = output[field]
    return example
def encoding_with_rubric_pair(example, tokenizer):
    # for sequence pair classification with rubric
    output = tokenizer(
        example["answer"],
        example["rubric"],) 
    for field in output:
        example[field] = output[field]
    return example
class Alice_Dataset:
    def __init__(self, enc_fn=encoding):
        self.enc_fn = enc_fn
        self.train = Dataset.from_csv(path_train)
        self.test_ua = Dataset.from_csv(path_ua)
        self.test_uq = Dataset.from_csv(path_uq)
        self.train, self.val = self.train.train_test_split(test_size=0.1, seed=8964).values()
    def get_encoding(self, tokenizer):
        self.train = self.train.map(lambda x: self.enc_fn(x, tokenizer), batched=True)
        self.val = self.val.map(lambda x: self.enc_fn(x, tokenizer), batched=True)
        self.test_ua = self.test_ua.map(lambda x: self.enc_fn(x, tokenizer), batched=True)
        self.test_uq = self.test_uq.map(lambda x: self.enc_fn(x, tokenizer), batched=True)
    def merge_scores(self,score="low"):
        """
        convert the selected level to 1 and the rest to 0. This is for binary clasification 
        where the model learns to differentiate between the selected level and the rest.
        """
        def _merge_scores(example):
            rubric = example["rubric"]
            rubric = json.loads(rubric)
            max_score = len(rubric) - 1
            if score == "low":
                target_score = 0
            elif score == "high":
                target_score = max_score
            elif score == "mid":
                target_score = max_score // 2
            example["label_id"] = 1 if example["level"] == target_score else 0
            return example
        self.train = self.train.map(lambda x: _merge_scores(x))
        self.val = self.val.map(lambda x: _merge_scores(x))
        self.test_ua = self.test_ua.map(lambda x: _merge_scores(x))
        self.test_uq = self.test_uq.map(lambda x: _merge_scores(x))
    @staticmethod
    def collate_fn(input_batch):
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids"]) for x in input_batch], batch_first=True)
        attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["attention_mask"]) for x in input_batch], batch_first=True)
        if "token_type_ids" in input_batch[0]:

            token_type_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["token_type_ids"]) for x in input_batch], batch_first=True)
        else: 
            token_type_ids = None
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label_id": torch.tensor([x["label_id"] for x in input_batch]),
        } 
        meta = {
            "sid": [x["sid"] for x in input_batch],
            "question_id": [x["question_id"] for x in input_batch],
        }
        return batch, meta 

class Alice_Rubric_Dataset(Alice_Dataset):
    def __init__(self, enc_fn=encoding_with_rubric_pair):
        super().__init__(enc_fn=enc_fn)
        self.expand_with_rubric()
    def expand_with_rubric(self):
        def _expand_dataset(dataset):
            expanded_data = []
            for example in dataset:
                rubric = json.loads(example["rubric"])
                for level, rb in rubric.items():
                    if not isinstance(rb, str):
                        print(rubric)
                    new_example = example.copy()
                    new_example["rubric"] = rb
                    new_example["rubric_level"] = int(level)  
                    new_example["label_id"] = 1 if new_example["level"] == int(level) else 0
                    expanded_data.append(new_example)
            expanded_data = Dataset.from_list(expanded_data)
            return expanded_data
        self.train = _expand_dataset(self.train)

    @staticmethod
    def collate_fn(input_batch):
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids"]) for x in input_batch], batch_first=True)
        attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["attention_mask"]) for x in input_batch], batch_first=True)
        if "token_type_ids" in input_batch[0]:
            token_type_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["token_type_ids"]) for x in input_batch], batch_first=True)
        else:
            token_type_ids = None
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label_id": torch.tensor([x["label_id"] for x in input_batch]),
        }
        meta = {
            "sid": [x["sid"] for x in input_batch],
            "question_id": [x["question_id"] for x in input_batch],
            "rubric_level": [x["rubric_level"] for x in input_batch],
            "level": [x["level"] for x in input_batch],
        }
        return batch, meta
if __name__ == "__main__":
    enable_caching()
    alice_ds = Alice_Rubric_Dataset()
