from typing import Literal
from datasets import enable_caching, Dataset
import json
import pandas as pd
from transformers import AutoTokenizer
"""
Dataprep pipeline: 
1. Load the Alice dataset from csv files.
2. Encode the dataset using the provided encoding functions for different model settings.
3. Provide collate functions for batching the dataset.
"""
path_train = "asap-sas-data/train.csv"
path_test = "asap-sas-data/test.csv"

RUBRICS = {}
for i in range(1,11):
    path_rub = f"asap-sas-data/rubrics/set{i}.json"
    with open(path_rub, "r") as f:
        RUBRICS[i] = json.load(f)["rubrics"]
# encoding functions 
def get_tokenizer(base_model: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(base_model)
    if "llama" in base_model.lower():
        tok.padding_side = "right"  
        tok.pad_token = tok.eos_token  # Ensure pad_token is set
    tok.sep_token = tok.sep_token or tok.eos_token  # Ensure sep_token is set
    return tok
def encode_rubric_pair(example, tokenizer):
    """
    Encode rubric and answer as a sequence pair.
    """
    output = tokenizer(example["answer"], example["rubric"], max_length=512, truncation=True)
    for field in output:
        example[field] = output[field]
    return example

def encode_with_fields(example, tokenizer, fields: list[str] = ["answer","rubric"], add_instruction: bool = False, format: Literal["natural_lang", "structured"] = "natural_lang"):
    """
    Encode the fields of the example using the tokenizer with natural language.
    Available fields: answer, question, sample_solution, rubric.
    """
    text2encode = ""
    for field in fields:
        if field not in example:
            raise ValueError(f"Field '{field}' not found in the example.")
        if format == "natural_lang":
            text2encode += f"{field}: {example[field]}\n"
        elif format == "structured":
            text2encode += f"<{field}>{example[field]}</{field}>\n"
    if add_instruction:
        text2encode = "Determine whether the rubric is satisfied by the answer:\n" + text2encode
    output = tokenizer(text2encode, max_length=512, truncation=True)
    for field in output:
        example[field] = output[field]
    return example
def encode_dataset(dataset, tokenizer, enc_fn, *args, **kwargs):
    dataset = dataset.map(lambda x: enc_fn(x, tokenizer, *args, **kwargs))
    return dataset
def group_by_id(dataset):
    """
    Group the expanded dataset back by id, reshaping input_ids, attention_mask etc 
    to have shape [R, S] where R is the number of rubrics and S is the sequence length.
    
    Args:
        dataset: Dataset that has been expanded with rubrics and encoded
        
    Returns:
        Dataset: Grouped dataset where each example contains tensors of shape [R, S]
    """
    # Group examples by their base ID (removing rubric suffixes if any)
    grouped_data = {}
    
    for example in dataset:
        base_id = example["id"]
        # Remove rubric expansion suffixes like "_ke0", "_sk1" etc if present
        # Keep the base structure for grouping
        if base_id not in grouped_data:
            grouped_data[base_id] = []
        grouped_data[base_id].append(example)
    
    # Convert grouped data back to list format
    regrouped_examples = []
    
    for base_id, examples in grouped_data.items():
        # Sort examples by rubric_level to ensure consistent ordering
        examples = sorted(examples, key=lambda x: x["rubric_level"])
        
        # Create the new grouped example
        grouped_example = {
            "id": base_id,
            "question_id": examples[0]["question_id"],
            "level": examples[0]["level"],  # Original level
            "num_rubrics": examples[0]["num_rubrics"],
            "rubric_level": examples[0]["rubric_level"]
        }
        
        # Stack the encoded fields
        grouped_example["input_ids"] = [ex["input_ids"] for ex in examples]
        grouped_example["attention_mask"] = [ex["attention_mask"] for ex in examples]
        
        # Handle token_type_ids if present
        if "token_type_ids" in examples[0]:
            grouped_example["token_type_ids"] = [ex["token_type_ids"] for ex in examples]
        
        # Keep labels and rubric levels for each rubric
        grouped_example["labels"] = examples[0]["level"]
        # Keep other metadata from first example
        for key in examples[0]:
            if key not in ["input_ids", "attention_mask", "token_type_ids", "labels", "num_rubrics","rubric_level"]:
                grouped_example[key] = examples[0][key]
        
        regrouped_examples.append(grouped_example)
    
    return Dataset.from_list(regrouped_examples)
class BaseLoader:
    """
    Load the splits of Alice dataset.
    """
    def __init__(self, train_frac=1):
        assert train_frac <= 1 and train_frac > 0, "train_frac must be in (0, 1]"
        self.train_frac = train_frac
        self.train = Dataset.from_pandas(pd.read_csv(path_train))
        self.test = Dataset.from_pandas(pd.read_csv(path_test))
        train_val_split = self.train.train_test_split(test_size=0.1, seed=42)
        self.val = train_val_split['test']
        if train_frac < 1:
            self.train = train_val_split['train'].train_test_split(test_size=1 - train_frac, seed=42)['train']
        else:
            self.train = train_val_split['train']
        self.train = self.train.map(self.retrieve_rubric)
        self.val = self.val.map(self.retrieve_rubric)
        self.test = self.test.map(self.retrieve_rubric)
    def retrieve_rubric(self, example):
        set_id = int(example["question_id"])
        rubrics = RUBRICS[set_id]
        rubrics = dict(sorted(rubrics.items(), key=lambda item: int(item[0])))
        rubrics = list(rubrics.values())
        example["rubric"] = rubrics
        example["num_rubrics"] = len(rubrics)
        return example
    def encode_all_splits(self,tokenizer,enc_fn, *args, **kwargs):
        self.train = encode_dataset(self.train, tokenizer, enc_fn, *args, **kwargs)
        self.val = encode_dataset(self.val, tokenizer, enc_fn, *args, **kwargs)
        self.test = encode_dataset(self.test, tokenizer, enc_fn, *args, **kwargs)


class RubricRetrievalLoader(BaseLoader):
    def __init__(self, train_frac=1):
        """
        Alice dataset for snet and xnet pair-wise ranking. 
        Each entry is expended to include all rubric levels.
        The labels is 1 if the level matches the rubric level, otherwise 0.
        """
        super().__init__(train_frac=train_frac)
 
    def expand_with_rubric(self):
        def _expand_dataset(dataset):
            expanded_data = []
            for example in dataset:
                rubric = example["rubric"]
                for level, rb in enumerate(rubric):
                    new_example = example.copy()
                    new_example["rubric"] = rb
                    new_example["rubric_level"] = int(level)
                    new_example["labels"] = 1 if int(new_example["level"]) == int(level) else 0
                    expanded_data.append(new_example)
            expanded_data = Dataset.from_list(expanded_data)
            return expanded_data
        self.train = _expand_dataset(self.train)
        self.val = _expand_dataset(self.val)
        self.test = _expand_dataset(self.test)

if __name__ == "__main__":
    from collate import xnet_collate_fn
    from torch.utils.data import DataLoader
    loader = RubricRetrievalLoader(train_frac=0.01) 
    loader.expand_with_rubric()
    tokenizer = get_tokenizer("bert-base-uncased")
    loader.encode_all_splits(tokenizer, encode_rubric_pair)
    loader.train = group_by_id(loader.train)

    train_dataloader = DataLoader(loader.train, batch_size=2, collate_fn=lambda x: xnet_collate_fn(x, pad_id=tokenizer.pad_token_id))
    for batch in train_dataloader:
        print(batch["input_ids"].shape)  # Should be [B, R, S]
        break