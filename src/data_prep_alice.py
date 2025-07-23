from typing import Literal
from datasets import load_dataset, enable_caching, Dataset, disable_caching
import json
import torch
from transformers import AutoTokenizer
"""
Dataprep pipeline: 
1. Load the Alice dataset from csv files.
2. Encode the dataset using the provided encoding functions for different model settings.
3. Provide collate functions for batching the dataset.
"""
enable_caching()
path_train = "alice_data/train.csv"
path_ua = "alice_data/test_ua.csv"
path_uq = "alice_data/test_uq.csv"
def get_tokenizer(base_model: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(base_model)
    if "llama" in base_model.lower():
        tok.padding_side = "right"
        tok.pad_token = tok.eos_token  # Ensure pad_token is set
    tok.sep_token = tok.sep_token or tok.eos_token  # Ensure sep_token is set
    return tok
def basic_encode(example, tokenizer):
    # encode answer only  
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
    """
    Encode rubric and answer as a sequence pair.
    """
    output = tokenizer(example["answer"], example["rubric"], max_length=512, truncation=True)
    for field in output:
        example[field] = output[field]
    return example
def encode_with_fields(example, tokenizer, fields: list[str] = ["answer","rubric"], add_instruction: bool = False):
    """
    Encode the fields of the example using the tokenizer with natural language.
    """
    text2encode = ""
    for field in fields:
        if field not in example:
            continue
        text2encode += f"{field.capitalize()}: {example[field]} "
    if add_instruction:
        text2encode = "Determine if rubric is satisfied by the answer. " + text2encode
    output = tokenizer(text2encode, max_length=512, truncation=True)
    for field in output:
        example[field] = output[field]
    return example

def encode_rubric_separate(example, tokenizer):
    """
    Encode rubric and answer separately into different keys for SNet 
    """
    answer_output = tokenizer(example["answer"], max_length=512, truncation=True)
    rubric_output = tokenizer(example["rubric"], max_length=512, truncation=True)
    for field in answer_output:
        example[field] = answer_output[field]
    for field in rubric_output:
        example[f"rubric_{field}"] = rubric_output[field]
    return example


class BaseLoader:
    """
    Load the splits of Alice dataset.
    """
    def __init__(self, train_frac=1):
        assert train_frac <= 1 and train_frac > 0, "train_frac must be in (0, 1]"
        self.train = Dataset.from_csv(path_train)
        if train_frac < 1:
            self.train = self.train.train_test_split(test_size=1-train_frac, seed=42)["train"]
        self.test_ua = Dataset.from_csv(path_ua)
        self.test_uq = Dataset.from_csv(path_uq)
        self.train, self.val = self.train.train_test_split(test_size=0.1, seed=8964).values()
    def get_encoding(self, tokenizer, enc_fn, *args, **kwargs):
        self.train = self.train.map(lambda x: enc_fn(x, tokenizer, *args, **kwargs))
        self.val = self.val.map(lambda x: enc_fn(x, tokenizer, *args, **kwargs))
        self.test_ua = self.test_ua.map(lambda x: enc_fn(x, tokenizer, *args, **kwargs))
        self.test_uq = self.test_uq.map(lambda x: enc_fn(x, tokenizer, *args, **kwargs))
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

class RubricRetrievalLoader(BaseLoader):
    def __init__(self, train_frac=1, input_fields: list[str] = ["answer","rubric"]):
        """
        Alice dataset for sbert and cross-ecoder pair-wise ranking. 
        Each entry is expended to include all rubric levels.
        The label_id is 1 if the level matches the rubric level, otherwise 0.
        """
        super().__init__(train_frac=train_frac)
        self.expand_with_rubric()
        self.input_fields = input_fields
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
    def collate_fn(input_batch, tokenizer=None):
        pad_ids = 0
        if tokenizer:
            pad_ids = tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids"]) for x in input_batch], batch_first=True, padding_value=pad_ids)
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
    def collate_rubric_seperate(input_batch, tokenizer=None):
        """
        colllate function for settings where the rubric and answer are encoded separately.
        """
        pad_ids = 0 
        if tokenizer:
            pad_ids = tokenizer.pad_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids"]) for x in input_batch], batch_first=True, padding_value=pad_ids)
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

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    loader = RubricRetrievalLoader(train_frac=0.1, input_fields=["answer", "rubric"])
    tokenizer = get_tokenizer("bert-base-uncased")
    loader.get_encoding(tokenizer, encode_with_fields, fields=loader.input_fields)
    train_loader = DataLoader(loader.train, batch_size=2, collate_fn=loader.collate_fn)
    for batch, meta in train_loader:
        print("Batch:")
        print(batch)
        print("\nDecoded tokens:")
        for i, input_ids in enumerate(batch["input_ids"]):
            tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())
            print(f"Sample {i}: {tokens}")
        break  # Just to test the first batch
