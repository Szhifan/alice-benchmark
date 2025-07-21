from datasets import load_dataset, enable_caching, Dataset,disable_caching
import json
import os 
import torch
from transformers import AutoTokenizer
disable_caching()
RUBRICS = []
rubrics_dir = "asap-sas-data/rubrics"
train_path = "asap-sas-data/train.csv"
test_path = "asap-sas-data/test.csv"
for i in range(1,11):
    rubrics_path = os.path.join(rubrics_dir, f"set{i}.json")
    with open(rubrics_path, "r") as f:
        rubrics = json.load(f)
        RUBRICS.append(rubrics)
def basic_encode(example, tokenizer):
    # basic encoding 
    output = tokenizer(
        example["EssayText"]
    )
    for field in output:
        example[field] = output[field]
    return example
def encode_rubric_pair(example, tokenizer):
    # for sequence pair classification
    output = tokenizer(
        example["EssayText"],
        example["rubric"],) 
    for field in output:
        example[field] = output[field]
    return example

def encode_rubric_prompt(example, tokenizer):
    # for prompt-based classification, can be used for T5 or other LLMs. 
    prompt = f"Essay: {example['EssayText']}. Rubric: {example['rubric']}. The essay meets the rubric."
    output = tokenizer(
        prompt
    )
    for field in output:
        example[field] = output[field]
    return example

def encode_rubric_span(example, tokenizer):
    """
    Encode answer + question + all the rubrics, seperated by the tokenizer's sep_token.
    """
    answer = example["EssayText"]
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
    qa_tokens = tokenizer.tokenize(" Answer: " + answer)
    tokens.extend(qa_tokens)
    answer_end = len(tokens)
    enc = tokenizer(tokens, is_split_into_words=True, max_length=512, truncation=True)
    for field in enc:
        example[field] = enc[field]
    example["rubric_indeces"] = rubric_indeces
    example["answer_span"] = (answers_start, answer_end)
    return example
class AsapDataset:
    def __init__(self,enc_fn=basic_encode):
        self.train = Dataset.from_csv(train_path)
        self.test = Dataset.from_csv(test_path)
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
            rubric_target = RUBRICS[int(example["EssaySet"]) - 1]["rubrics"][str(target_score)]
            example["rubrics"] = rubric_target
            return example

        self.train = self.train.map(lambda x: _merge_scores(x))
        self.val = self.val.map(lambda x: _merge_scores(x))
        self.test = self.test.map(lambda x: _merge_scores(x))
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
            "id": [x["Id"] for x in input_batch],
            "EssaySet": [x["EssaySet"] for x in input_batch],
            "answer": [x["EssayText"] for x in input_batch]
        }
        return batch, meta 

class AsapRubric(AsapDataset):
    def __init__(self, enc_fn=encode_rubric_pair):
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
                    new_entry["rubric_level"] = int(key) 
                    new_entry["label_id"] = 1 if entry["score"] == new_entry["rubric_level"] else 0
                    new_dataset.append(new_entry)
            new_dataset = Dataset.from_list(new_dataset)
            return new_dataset

        self.train = _expand_dataset(self.train)
        self.val = _expand_dataset(self.val)
        self.test = _expand_dataset(self.test)
    @staticmethod
    def collate_fn(input_batch):
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["input_ids"]) for x in input_batch], batch_first=True
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor(x["attention_mask"]) for x in input_batch], batch_first=True
        )
        if "token_type_ids" in input_batch[0]:
            token_type_ids = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x["token_type_ids"]) for x in input_batch], batch_first=True
            )
        else:
            token_type_ids = None

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "label_id": torch.tensor([x["label_id"] for x in input_batch]),
        }

        meta = {
            "id": [x["Id"] for x in input_batch],
            "EssaySet": [x["EssaySet"] for x in input_batch],
            "answer": [x["EssayText"] for x in input_batch],
            "rubric_level": [x["rubric_level"] for x in input_batch],
            "level": [x["score"] for x in input_batch],
        }

        return batch, meta


class AsapRubricPointer(AsapDataset):
    def __init__(self, enc_fn=encode_rubric_span):
        super().__init__(enc_fn=enc_fn)
        self.add_rubric()
    def add_rubric(self):
        def _add_rubric(example):
            essay_set = int(example["EssaySet"]) - 1
            example["rubric"] = [rub for rub in RUBRICS[essay_set]["rubrics"].values()]
            return example
        self.train = self.train.map(lambda x: _add_rubric(x))
        self.val = self.val.map(lambda x: _add_rubric(x))
        self.test = self.test.map(lambda x: _add_rubric(x))
    @staticmethod
    def collate_fn(input_batch):
        batch = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x["input_ids"]) for x in input_batch], batch_first=True
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(x["attention_mask"]) for x in input_batch], batch_first=True,
            ),
            "answer_span": torch.tensor([x["answer_span"] for x in input_batch], dtype=torch.long),
            }
        max_rubrics = max(len(x["rubric"]) for x in input_batch)
        rubric_span_tensor = torch.zeros(len(input_batch), max_rubrics, 2, dtype=torch.long)
        rubric_mask_tensor = torch.zeros(len(input_batch), max_rubrics, dtype=torch.bool)
        for i, example in enumerate(input_batch):
            for j, (start, end) in enumerate(example["rubric_indeces"]):
                rubric_span_tensor[i, j, 0] = start
                rubric_span_tensor[i, j, 1] = end
                rubric_mask_tensor[i, j] = True
        batch["rubric_span"] = rubric_span_tensor
        batch["rubric_mask"] = rubric_mask_tensor
        batch["label_id"] = torch.tensor([x["score"] for x in input_batch])
        meta = {
            "answer": [x["EssayText"] for x in input_batch],
        }
        return batch, meta

if __name__ == "__main__":
    # Example usage
    from torch.utils.data import DataLoader
    dts = AsapRubricPointer()
    dts.get_encoding(AutoTokenizer.from_pretrained("bert-base-uncased"))
    tr_loader = DataLoader(
        dts.train, batch_size=16, shuffle=True, collate_fn=dts.collate_fn
    )
    print(len(tr_loader))
    