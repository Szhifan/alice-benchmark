from typing import Literal
from datasets import load_dataset, enable_caching, Dataset,disable_caching
import json
import torch
from transformers import AutoTokenizer
"""
Dataprep pipeline: 
1. Load the Alice dataset from csv files.
2. Encode the dataset using the provided encoding functions for different model settings.
3. Provide collate functions for batching the dataset.
"""
disable_caching()
path_train = "alice_data/ALICE_train_new.csv"
path_ua = "alice_data/ALICE_UA_new.csv"
path_uq = "alice_data/ALICE_UQ_new.csv"
def basic_encode(example, tokenizer):
    # basic encoding 
    output = tokenizer(example["answer"], max_length=512, truncation=True)
    for field in output:
        example[field] = output[field]
    return example

def encode_solution_pair(example, tokenizer):
    # encode answer and sample solution as a sequence pair
    output = tokenizer(example["sample_solution"], example["answer"], max_length=512, truncation=True) 
    for field in output:
        example[field] = output[field]
    return example

def encode_rubric_pair(example, tokenizer):
    # for sequence pair classification with rubric
    sep_tok = tokenizer.sep_token if tokenizer.sep_token else tokenizer.eos_token  # for t5
    structured_input = f"{sep_tok}".join([example["answer"], example["rubric"]])
    nl_input = f"Answer: {example['answer']}. Rubric: {example['rubric']}." 

    output = tokenizer(structured_input, max_length=512, truncation=True)
    for field in output:
        example[field] = output[field]
    return example
def encode_rubric_separate(example, tokenizer):
    """
    Encode rubric and answer separately into different keys for double encoder models.
    """
    answer_output = tokenizer(example["answer"], max_length=512, truncation=True)
    rubric_output = tokenizer(example["rubric"], max_length=512, truncation=True)
    for field in answer_output:
        example[field] = answer_output[field]
    for field in rubric_output:
        example[f"rubric_{field}"] = rubric_output[field]
    return example
def encode_rubric_solution_pair(example, tokenizer):
    """
    Encode answer + rubric + sample_solution, seperated by the tokenizer's sep_token.
    """
    sep_tok = tokenizer.sep_token if tokenizer.sep_token else tokenizer.eos_token  # for t5
    structured_input = f"{sep_tok}".join([example["answer"], example["sample_solution"], example["rubric"]])
    nl_input = f"Answer: {example['answer']}. Sample solution: {example['sample_solution']}. Rubric: {example['rubric']}. Does the rubric match the answer?"
    output = tokenizer(structured_input, max_length=512, truncation=True)
    for field in output:
        example[field] = output[field]
    return example
def encode_rubric_span(example, tokenizer):
    """
    Grasp encoding 
    """
    answer = example["answer"]
    question = example["question"]
    rubrics = example["rubric"]
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    tokens = [cls_token]
    rubric_indeces = []
    for rubric in rubrics:
        start = len(tokens)
        rub_tokens = tokenizer.tokenize(rubric) + [sep_token]
        tokens.extend(rub_tokens)
        end = len(tokens)
        rubric_indeces.append((start, end))
    answers_start = len(tokens)
    # qa_tokens = tokenizer.tokenize("Question: " + question + " Answer: " + answer)
    qa_tokens = tokenizer.tokenize(answer)
    tokens.extend(qa_tokens)
    answer_end = len(tokens)
    enc = tokenizer(tokens, is_split_into_words=True)
    for field in enc:
        example[field] = enc[field]
    example["rubric_indeces"] = rubric_indeces
    
    example["answer_span"] = (answers_start, answer_end)
    return example
def encode_rubric_span_separate(example, tokenizer):
    """
    Grasp encoding with separate encoding for answer and rubric.
    """
    answer_output = tokenizer(example["answer"], example["question"], max_length=512, truncation=True)
    rubrics = example["rubric"]
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    rubric_tokens = [cls_token]
    rubric_indeces = []
    for rubric in rubrics:
        start = len(rubric_tokens)
        rub_tokens = tokenizer.tokenize(rubric) + [sep_token]
        rubric_tokens.extend(rub_tokens)
        end = len(rubric_tokens)
        rubric_indeces.append((start, end))
    rubric_output = tokenizer(rubric_tokens, is_split_into_words=True, max_length=512, truncation=True)
    for field in answer_output:
        example[field] = answer_output[field]
    for field in rubric_output:
        example[f"rubric_{field}"] = rubric_output[field]
    example["rubric_indeces"] = rubric_indeces
    return example
class AliceDataset:
    """
    Base alice dataset.
    """
    def __init__(self, enc_fn=basic_encode,train_frac=1):
        assert train_frac <= 1 and train_frac > 0, "train_frac must be in (0, 1]"
        self.enc_fn = enc_fn
        self.train = Dataset.from_csv(path_train)
        if train_frac < 1:
            self.train = self.train.train_test_split(test_size=1-train_frac, seed=42)["train"]
        self.test_ua = Dataset.from_csv(path_ua)
        self.test_uq = Dataset.from_csv(path_uq)
        self.train, self.val = self.train.train_test_split(test_size=0.1, seed=8964).values()
    def get_encoding(self, tokenizer):
        self.train = self.train.map(lambda x: self.enc_fn(x, tokenizer))
        self.val = self.val.map(lambda x: self.enc_fn(x, tokenizer))
        self.test_ua = self.test_ua.map(lambda x: self.enc_fn(x, tokenizer))
        self.test_uq = self.test_uq.map(lambda x: self.enc_fn(x, tokenizer))
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
            "id": [x["sid"] for x in input_batch],
            "answer": [x["answer"] for x in input_batch],
        }
        return batch, meta 

class AliceRubricDataset(AliceDataset):
    def __init__(self, enc_fn=encode_rubric_pair, train_frac=1):
        """
        Alice datast for sbert and cross-ecoder pair-wise ranking. 
        Each entry is expended to include all rubric levels.
        The label_id is 1 if the level matches the rubric level, otherwise 0.
        """
        super().__init__(enc_fn=enc_fn, train_frac=train_frac)
        self.expand_with_rubric()
    def expand_with_rubric(self):
        def _expand_dataset(dataset):
            expanded_data = []
            for example in dataset:
                rubric = json.loads(example["rubric"])
                for level, rb in rubric.items():
                    new_example = example.copy()
                    new_example["rubric"] = rb
                    new_example["rubric_level"] = int(level)  
                    new_example["label_id"] = 1 if new_example["level"] == int(level) else 0
                    expanded_data.append(new_example)
            expanded_data = Dataset.from_list(expanded_data)
            return expanded_data
        self.train = _expand_dataset(self.train)
        self.val = _expand_dataset(self.val)
        self.test_ua = _expand_dataset(self.test_ua)
        self.test_uq = _expand_dataset(self.test_uq)

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
            "id": [x["id"] for x in input_batch],
            "rubric_level": [x["rubric_level"] for x in input_batch],
            "level": [x["level"] for x in input_batch],
        }
        return batch, meta
    @staticmethod
    def collate_rubric_seperate(input_batch):
        """
        colllate function for settings where the rubric and answer are encoded separately.
        """
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids"]) for x in input_batch], batch_first=True)
        attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["attention_mask"]) for x in input_batch], batch_first=True)

        rubric_input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["rubric_input_ids"]) for x in input_batch], batch_first=True)
        rubric_attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["rubric_attention_mask"]) for x in input_batch], batch_first=True)

        
        batch = {
            "input_ids_a": input_ids,
            "attention_mask_a": attention_mask,
            "input_ids_b": rubric_input_ids,
            "attention_mask_b": rubric_attention_mask,
            "label_id": torch.tensor([x["label_id"] for x in input_batch]),
        }

        meta = {
            "id": [x["id"] for x in input_batch],
            "rubric_level": [x["rubric_level"] for x in input_batch],
            "level": [x["level"] for x in input_batch],
        }
        return batch, meta
class AliceRubricPointer(AliceDataset):
    def __init__(self, enc_fn=encode_rubric_span, train_frac=1):
        super().__init__(enc_fn=enc_fn, train_frac=train_frac)
        self.tranform_rubric()
    def tranform_rubric(self):
        def _transform(example):
            rubric = json.loads(example["rubric"])
            example["rubric"] = [rub for rub in rubric.values()]
             
            return example
        self.train = self.train.map(lambda x: _transform(x))
        self.val = self.val.map(lambda x: _transform(x))
        self.test_ua = self.test_ua.map(lambda x: _transform(x))
        self.test_uq = self.test_uq.map(lambda x: _transform(x))
    @staticmethod
    def collate_fn(input_batch):
        batch = {
        "input_ids": torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids"]) for x in input_batch], batch_first=True),
        "attention_mask": torch.nn.utils.rnn.pad_sequence([torch.tensor(x["attention_mask"]) for x in input_batch], batch_first=True),
        "label_id": torch.tensor([x["level"] for x in input_batch]),
        }
        max_rubrics = max(len(x["rubric"]) for x in input_batch)
        span_tensor = torch.zeros((len(input_batch), max_rubrics, 2), dtype=torch.long)
        mask_tensor = torch.zeros((len(input_batch), max_rubrics), dtype=torch.bool)
        for i, example in enumerate(input_batch):
            for j, (start, end) in enumerate(example["rubric_indeces"]):
                span_tensor[i, j, 0] = start
                span_tensor[i, j, 1] = end
                mask_tensor[i, j] = True
        batch["rubric_span"] = span_tensor
        batch["rubric_mask"] = mask_tensor
        batch["answer_span"] = torch.tensor([example["answer_span"] for example in input_batch], dtype=torch.long)
        meta = {"id": [x["sid"] for x in input_batch],
                "answer": [x["answer"] for x in input_batch],
          }
        return batch, meta
    @staticmethod
    def collate_rubric_separate(input_batch):   
        """
        colllate function for settings where the rubric and answer are encoded separately.
        """
        batch = {
            "ans_input_ids": torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids"]) for x in input_batch], batch_first=True),
            "ans_attention_mask": torch.nn.utils.rnn.pad_sequence([torch.tensor(x["attention_mask"]) for x in input_batch], batch_first=True),
            "rubric_input_ids": torch.nn.utils.rnn.pad_sequence([torch.tensor(x["rubric_input_ids"]) for x in input_batch], batch_first=True),
            "rubric_attention_mask": torch.nn.utils.rnn.pad_sequence([torch.tensor(x["rubric_attention_mask"]) for x in input_batch], batch_first=True),
            "label_id": torch.tensor([x["level"] for x in input_batch]),
        }
        max_rubrics = max(len(x["rubric_indeces"]) for x in input_batch)
        span_tensor = torch.zeros((len(input_batch), max_rubrics, 2), dtype=torch.long)
        mask_tensor = torch.zeros((len(input_batch), max_rubrics), dtype=torch.bool)
        for i, example in enumerate(input_batch):
            for j, (start, end) in enumerate(example["rubric_indeces"]):
                span_tensor[i, j, 0] = start
                span_tensor[i, j, 1] = end
                mask_tensor[i, j] = True
        batch["rubric_span"] = span_tensor
        batch["rubric_mask"] = mask_tensor
        meta = {
            "id": [x["sid"] for x in input_batch],
            "answer": [x["answer"] for x in input_batch],
        }
        return batch, meta
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    import numpy as np
    dts = AliceRubricDataset(enc_fn=encode_rubric_pair)
    tok = AutoTokenizer.from_pretrained("google-t5/t5-small")
    dts.get_encoding(tok)
    input_ids = dts.train[0]["input_ids"]
    print(tok.decode(input_ids, skip_special_tokens=False))