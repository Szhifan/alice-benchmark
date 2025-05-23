from datasets import load_dataset 
from typing import Literal
from datasets import load_dataset, enable_caching, Dataset, disable_progress_bars
import json
import pandas as pd
import os 
import torch
from transformers import AutoTokenizer
disable_progress_bars()
raw_fields = ["Id","EssaySet","score","EssayText"]
path_train = "data/train.tsv"
path_test = "data/public_leaderboard.tsv"
path_test_score = "data/public_leaderboard_solution.csv"
train_df = pd.read_csv(path_train, sep="\t")
train_df = train_df.rename(columns={"Score1": "score"})
train_df = train_df.drop(columns=["Score2"])

test_df = pd.read_csv(path_test, sep="\t")
test_score_df = pd.read_csv(path_test_score)
test_df = pd.concat([test_df, test_score_df], axis=1)
test_df = test_df.drop(columns=["id", "essay_set", "Usage"])
test_df = test_df.rename(columns={"essay_score": "score"}) 
RUBRICS = []
rubrics_dir = "rubrics/"
for i in range(1,11):
    rubrics_path = os.path.join(rubrics_dir, f"set{i}.json")
    with open(rubrics_path, "r") as f:
        rubrics = json.load(f)
        RUBRICS.append(rubrics)
def encoding(example, tokenizer):
    # basic encoding 
    output = tokenizer(
        example["EssayText"]
    )
    for field in output:
        example[field] = output[field]
    return example
def encoding_with_rubric_pair(example, tokenizer):
    # for sequence pair classification
    output = tokenizer(
        example["EssayText"],
        example["rubric"],) 
    for field in output:
        example[field] = output[field]
    return example

def encoding_with_rubric_prompt(example, tokenizer):
    # for prompt-based classification, can be used for T5 or other LLMs. 
    prompt = f"Essay: {example['EssayText']}. Rubric: {example['rubric']}. The essay meets the rubric."
    output = tokenizer(
        prompt
    )
    for field in output:
        example[field] = output[field]
    return example
def encoding_for_conditional_generation(example, tokenizer):
    # for conditional generation task
    prompt_enc = f"Essay: {example['EssayText']} \n Rubric: {example['rubric']}"
    output_enc = tokenizer(
        prompt_enc,
    )
    if example["rubric_score"] == example["score"]:
        prompt_dec = "yes"
    else:
        prompt_dec = "no"
    output_dec = tokenizer(
        prompt_dec,
    )
    for field in output_enc:
        example[field] = output_enc[field]
    for field in output_dec:
        example[field + "_dec"] = output_dec[field]
    return example
class Asap_Dataset:
    def __init__(self,enc_fn=encoding):
        self.train = Dataset.from_pandas(train_df)
        self.test = Dataset.from_pandas(test_df)
        self.train, self.val = self.train.train_test_split(test_size=0.1,seed=42).values()
        self.enc_fn = enc_fn
    def get_encoding(self, tokenizer):
        self.train = self.train.map(
            lambda x: self.enc_fn(x, tokenizer)
        )
        self.val = self.val.map(
            lambda x: self.enc_fn(x, tokenizer)
        )
        self.test = self.test.map(
            lambda x: self.enc_fn(x, tokenizer)
        )
    def merge_scores(self,score="low"):
        def _merge_scores(example):
            essay_set_info = RUBRICS[int(example["EssaySet"]) - 1]["dataset_info"]
            max_score = max(essay_set_info["rubric_range"])
            if score == "low":
                target_score = 0 
            elif score == "high":
                target_score = max_score
            elif score == "mid":
                target_score = max_score - 1 
            if example["score"] == target_score:
                example["label_id"] = 1
            else:
                example["label_id"] = 0
            return example
            
        self.train = self.train.map(
            lambda x: _merge_scores(x),     
        )
        self.val = self.val.map(
            lambda x: _merge_scores(x)
        )
        self.test = self.test.map(
            lambda x: _merge_scores(x)
        )
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
            "Id": [x["Id"] for x in input_batch],
            "EssaySet": [x["EssaySet"] for x in input_batch],
            "EssayText": [x["EssayText"] for x in input_batch]
        }
        return batch, meta 

class Asap_Rubric(Asap_Dataset):
    def __init__(self, enc_fn=encoding_with_rubric_pair):
        super().__init__(enc_fn=enc_fn)
        self.expand_with_rubrics()
    
    def expand_with_rubrics(self):
        def _expand_dataset(dataset):

            new_dataset = []
            for entry in dataset: 
   
                essay_set = int(entry["EssaySet"]) 

                essay_rubrics = RUBRICS[essay_set - 1]["rubrics"]
                for key, rubric in essay_rubrics.items():
                    
                    new_entry = entry.copy()
                    new_entry["rubric"] = rubric
                    new_entry["rubric_score"] = int(key) 
                    new_dataset.append(new_entry)
            new_dataset = Dataset.from_list(new_dataset)
            return new_dataset

        self.train = _expand_dataset(self.train)
        self.val = _expand_dataset(self.val)
        self.test = _expand_dataset(self.test)


class Asap_Rubric_Conditional_Gen(Asap_Rubric):
    def __init__(self,enc_fn=encoding_for_conditional_generation):
        super().__init__(enc_fn=enc_fn)
        
    @staticmethod
    def collate_fn(input_batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["input_ids"]) for x in input_batch], batch_first=True
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["attention_mask"]) for x in input_batch], batch_first=True
        )


        decoder_input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["input_ids_dec"]) for x in input_batch], batch_first=True
        )
        decoder_attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["attention_mask_dec"]) for x in input_batch], batch_first=True
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        batch_decoder = {
            "input_ids": decoder_input_ids,
            "attention_mask": decoder_attention_mask,
        }

        meta = {
            "Id": [x["Id"] for x in input_batch],
            "EssaySet": [x["EssaySet"] for x in input_batch],
            "EssayText": [x["EssayText"] for x in input_batch],
        }

        return batch, batch_decoder, meta

if __name__ == "__main__":
    asap_r = Asap_Dataset()
    asap_r.merge_scores()
    train_ds = asap_r.train
    print(len(train_ds))
