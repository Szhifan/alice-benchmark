from typing import Literal
from datasets import enable_caching, Dataset, disable_caching
import json
import torch
from transformers import AutoTokenizer
import random
disable_caching()
"""
Dataprep pipeline: 
1. Load the Alice dataset from json files.
2. Encode the dataset using the provided encoding functions for different model settings.
3. Provide collate functions for batching the dataset.
"""
path_train = "alice_data/train.json"
path_ua = "alice_data/test_ua.json"
path_uq = "alice_data/test_uq.json"
FIELD_EN2DE = {
    "answer": "Antwort",
    "rubric": "Rubrik",
    "question": "Frage",
    "sample_solution": "Lösung"
}
path_meta_bio = "question_meta/bio.json"
path_meta_chemie = "question_meta/chemie.json"
path_meta_physik = "question_meta/physik.json"
path_meta_mathe = "question_meta/mathe.json"
with open(path_meta_bio, "r") as f:
    meta_bio = json.load(f)
with open(path_meta_chemie, "r") as f:
    meta_chemie = json.load(f)
with open(path_meta_physik, "r") as f:
    meta_physik = json.load(f)
with open(path_meta_mathe, "r") as f:
    meta_mathe = json.load(f)
def basic_encode(example, tokenizer):
    # encode answer only  
    output = tokenizer(example["answer"], max_length=512, truncation=True)
    for field in output:
        example[field] = output[field]
    example["labels"] = int(example["level"])
    return example

def encode_solution_pair(example, tokenizer):
    output = tokenizer(example["sample_solution"], example["answer"], max_length=512, truncation=True) 
    for field in output:
        example[field] = output[field]
    example["labels"] = int(example["level"])
    return example
def encode_rubric_pair(example, tokenizer):
    """
    For bert-like models.
    Encode rubric and answer as a sequence pair.
    """
    output = tokenizer(example["answer"], example["rubric"], max_length=512, truncation=True)
    for field in output:
        example[field] = output[field]
    return example
def encode_fields_special_tokens(example, tokenizer, fields: list[str] = ["answer","rubric"]): 
    """
    For bert-like models.
    Encode multiple fields with special tokens.
    """
    text2encode = []
    for field in fields: 
        if field not in example:
            raise ValueError(f"Field '{field}' not found in the example.")
        text2encode.append(example[field])

    text2encode = tokenizer.sep_token.join(text2encode)
    output = tokenizer(text2encode, max_length=512, truncation=True, add_special_tokens=True)
    return output
def encode_with_fields(example, tokenizer, fields: list[str] = ["answer","rubric"], add_instruction: bool = False, format: Literal["natural_lang", "structured"] = "natural_lang"):
    """
    For LLM models.
    Encode the fields of the example using the tokenizer with natural language.
    Available fields: answer, question, sample_solution, rubric.
    """
    text2encode = ""
    for field in fields:
        if field not in example:
            raise ValueError(f"Field '{field}' not found in the example.")
        if format == "natural_lang":
            text2encode += f"{FIELD_EN2DE[field]}: {example[field]}\n"
        elif format == "structured":
            text2encode += f"<{FIELD_EN2DE[field]}>{example[field]}</{FIELD_EN2DE[field]}>\n"
    if add_instruction:
        text2encode = "Bestimmen Sie, ob die Rubrik durch die Antwort erfüllt wird:\n" + text2encode
    output = tokenizer(text2encode, max_length=512, truncation=True)
    for field in output:
        example[field] = output[field]
    return example
def encode_special_tokens_snet(example, tokenizer, fields: list[str] = ["answer"]):
    """
    For Bert models.
    Encode rubric as one encoding and other fields as another encoding for SNet.
    """
    # Encode rubric separately
    rubric_output = tokenizer([rub for rub in example["rubric"]], max_length=512, truncation=True, add_special_tokens=True)
    for field in rubric_output:
        example[f"rubric_{field}"] = rubric_output[field]
    
    # Encode other fields together
    text2encode = []
    for field in fields:
        if field not in example:
            raise ValueError(f"Field '{field}' not found in the example.")
        text2encode.append(example[field])
    
    text2encode = tokenizer.sep_token.join(text2encode)
    other_output = tokenizer(text2encode, max_length=512, truncation=True, add_special_tokens=True)
    for field in other_output:
        example[field] = other_output[field]
    example["labels"] = int(example["level"])
    return example

def encode_with_fields_snet(
    example, tokenizer, fields: list[str] = ["answer"], 
    add_instruction: bool = False, format: Literal["natural_lang", "structured"] = "structured"
):
    """
    For LLM models.
    Encoding function for snet llm architecture.
    """
    rubric_encoded = tokenizer([rub for rub in example["rubric"]], max_length=512, truncation=True)
    query2encode = ""
    for field in fields:
        if field not in example:
            raise ValueError(f"Field '{field}' not found in the example.")
        if format == "natural_lang":
            query2encode += f"{FIELD_EN2DE[field]}: {example[field]}\n"
        elif format == "structured":
            query2encode += f"<{FIELD_EN2DE[field]}>{example[field]}</{FIELD_EN2DE[field]}>\n"
    if add_instruction:
        query2encode = "Bestimmen Sie, ob die Rubrik durch die Antwort erfüllt wird:\n" + query2encode
    query_output = tokenizer(query2encode, max_length=512, truncation=True)
    for field in query_output:
        example[field] = query_output[field]
    for field in rubric_encoded:
        example[f"rubric_{field}"] = rubric_encoded[field]
    example["labels"] = int(example["level"])
    return example


def encode_generation(example, tokenizer, train=True, additional_fields=None):
    """
    Encode text for generation task
    """
    rubric = example["rubric"]
    addition_input_text = ""
    if additional_fields is not None:
        for field in additional_fields:
            if field not in example:
                raise ValueError(f"Field '{field}' not found in the example.")
            addition_input_text += f"{FIELD_EN2DE[field]}: {example[field]}\n"
    
    rubric_text = [f"Niveau {key}: {value}" for key, value in rubric.items()]
    
    text2encode = f"""Aufgabe: Bewerten Sie die folgende Schülerantwort anhand der gegebenen Bewertungskriterien.
    {addition_input_text}
    Schülerantwort: {example['answer']}
    
    Bewertungskriterien:
    {chr(10).join(rubric_text)}
    Welches Bewertungsniveau entspricht dieser Antwort am besten?
    """
        
    if train:
        response = f"Antwort: Niveau {example['level']} {tokenizer.eos_token}"
        text2encode += response
    
    encoded = tokenizer(text2encode, max_length=1024, truncation=True)
    
    for field in encoded:
        example[field] = encoded[field]
    example["text"] = text2encode
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
        regrouped_examples.append(grouped_example)
    
    return Dataset.from_list(regrouped_examples)
class BaseLoader:
    """
    Load the splits of Alice dataset.
    """
    def __init__(self, train_frac=1, task_type="lp"):
        assert train_frac <= 1 and train_frac > 0, "train_frac must be in (0, 1]"
        assert task_type in ["lp","ke","sk"] , "task_type must be one of ['lp','ke','sk']"
        self.problematic_ids = []
        self.task_type = task_type
        self.train_frac = train_frac
        self.format_dataset()
        
    def retrieve_metadata(self,entry:dict):
        if "bio" in entry["id"]:
            meta = meta_bio
        elif "chemie" in entry["id"]:
            meta = meta_chemie
        elif "physik" in entry["id"]:
            meta = meta_physik
        elif "math" in entry["id"]:
            meta = meta_mathe
        else:
            raise ValueError(f"Unknown subject in id {entry['id']}")
        meta_info = meta[entry["question_id"]]
        fields_to_keep = ["id","question_id","question","answer","sample_solution","rubric","level", "num_rubrics"]
        if self.task_type == "lp":
            entry["question"] = meta_info.get("prompt", "")
            entry["sample_solution"] = meta_info.get("solution", "")
            rubric = meta_info["learning_performance"]
            entry["rubric"] = [v['rule'] for v in rubric.values()]
            entry["level"] = int(next(iter(entry["learning_performance"].values())))
            entry["num_rubrics"] = len(rubric)
            new_entry = {k: entry[k] for k in entry if k in fields_to_keep}
            return new_entry
        elif self.task_type == "ke":
            expending_entries = []
            if not entry.get("knowledge_elements"):
                return []
            for i, ke in enumerate(entry['knowledge_elements']):
                new_entry = entry.copy()
                new_entry["id"] = f"{entry['id']}_ke{i}"
                new_entry["question"] = meta_info.get("prompt", "")
                new_entry["sample_solution"] = meta_info.get("solution", "")
                ke_rubric  = meta_info.get("knowledge_elements", {}).get(ke, {})
                if len(ke_rubric) == 0:
                    continue    
                level_range = set(ke_rubric.keys())
                
                if str(entry["knowledge_elements"][ke]) not in level_range:
                    self.problematic_ids.append({
                        "id": new_entry["id"],
                        "type": "knowledge_elements",
                        "name": ke,
                        "field": "level",
                        "value": entry["knowledge_elements"][ke],
                        "range": level_range
                    })
                    continue
                # in some rubrics, levels are not continuous and should be normalized during training {0,1,3} -> {0,1,2}
                level_remap = {k:i for i,k in enumerate(sorted(level_range, key=float))} 
                level = level_remap[str(entry["knowledge_elements"][ke])]
                ke_rubric = [f"{ke}: {v['description']}" for v in ke_rubric.values()]
                new_entry["rubric"] = ke_rubric
                new_entry["level"] = level
                new_entry["num_rubrics"] = len(ke_rubric)
                new_entry = {k: new_entry[k] for k in new_entry if k in fields_to_keep}

                expending_entries.append(new_entry)
            return expending_entries
        elif self.task_type == "sk":
            if not entry.get("skills"):
                return []
            expending_entries = []
            for i, sk in enumerate(entry['skills']):
                
                new_entry = entry.copy()
                new_entry["id"] = f"{entry['id']}_sk{i}"
                new_entry["question"] = meta_info.get("prompt", "")
                new_entry["sample_solution"] = meta_info.get("solution", "")
                sk_rubric  = meta_info.get("skills", {}).get(sk, {})
                if len(sk_rubric) == 0:
                    continue
                level_range = set(sk_rubric.keys())

                if str(entry["skills"][sk]) not in level_range:
                    self.problematic_ids.append({"id": new_entry["id"], "type": "skills", "name": sk, "field": "level", "value": entry["skills"][sk], "range": level_range})
                    continue
                # in some rubrics, levels are not continuous and should be normalized during training {0,1,3} -> {0,1,2}
                level_remap = {k:i for i,k in enumerate(sorted(level_range, key=float))} 
                level = level_remap[str(entry["skills"][sk])]
                sk_rubric = [f"{sk}: {v['description']}" for v in sk_rubric.values()]
                new_entry["rubric"] = sk_rubric
                new_entry["level"] = level
                new_entry["num_rubrics"] = len(sk_rubric)
                new_entry = {k: new_entry[k] for k in new_entry if k in fields_to_keep}

                expending_entries.append(new_entry)
            return expending_entries

                

    def format_dataset(self):
        with open(path_train, "r") as f:
            train_data = json.load(f)
        with open(path_ua, "r") as f:
            test_ua_data = json.load(f)
        with open(path_uq, "r") as f:
            test_uq_data = json.load(f)
        new_train = []
        for entry in train_data:
            if self.task_type == "lp":
                new_entry = self.retrieve_metadata(entry)
                new_train.append(new_entry)
            else:
                for new_entry in self.retrieve_metadata(entry):
                    new_train.append(new_entry)
        train_data = new_train
        new_test_ua = []
        for entry in test_ua_data:
            if self.task_type == "lp":
                new_entry = self.retrieve_metadata(entry)
                new_test_ua.append(new_entry)
            else:
                for new_entry in self.retrieve_metadata(entry):
                    new_test_ua.append(new_entry)
        test_ua_data = new_test_ua
        new_test_uq = []
        for entry in test_uq_data:
            if self.task_type == "lp":
                new_entry = self.retrieve_metadata(entry)
                new_test_uq.append(new_entry)
            else:
                for new_entry in self.retrieve_metadata(entry):
                    new_test_uq.append(new_entry)
        test_uq_data = new_test_uq
        train_dataset = Dataset.from_list(train_data)
        if self.train_frac < 1:
            train_dataset = train_dataset.shuffle(seed=42).select(range(int(len(train_dataset)*self.train_frac)))
        split = train_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split["train"]
        val_dataset = split["test"]
        self.train = train_dataset
        self.val = val_dataset
        self.test_ua = Dataset.from_list(test_ua_data)
        self.test_uq = Dataset.from_list(test_uq_data)

    
    def encode_all_splits(self,tokenizer,enc_fn, *args, **kwargs):
        self.train = encode_dataset(self.train, tokenizer, enc_fn, *args, **kwargs)
        self.val = encode_dataset(self.val, tokenizer, enc_fn, *args, **kwargs)
        self.test_ua = encode_dataset(self.test_ua, tokenizer, enc_fn, *args, **kwargs)
        self.test_uq = encode_dataset(self.test_uq, tokenizer, enc_fn, *args, **kwargs)


class RubricRetrievalLoader(BaseLoader):
    def __init__(self, train_frac=1, task_type="lp"):
        """
        Alice dataset for snet and xnet pair-wise ranking. 
        Each entry is expended to include all rubric levels.
        The labels is 1 if the level matches the rubric level, otherwise 0.
        """
        super().__init__(train_frac=train_frac, task_type=task_type)
 
    def expand_with_rubric(self):
        def _expand_dataset(dataset):
            expanded_data = []
            for example in dataset:
                rubric = example["rubric"]
                for i, rb in enumerate(rubric):
                    new_example = example.copy()
                    new_example["rubric"] = rb
                    new_example["rubric_level"] = i
                    new_example["labels"] = 1 if int(new_example["level"]) == i else 0
                    expanded_data.append(new_example)
            expanded_data = Dataset.from_list(expanded_data)
            return expanded_data
        self.train = _expand_dataset(self.train)
        self.val = _expand_dataset(self.val)
        self.test_ua = _expand_dataset(self.test_ua)
        self.test_uq = _expand_dataset(self.test_uq)
if __name__ == "__main__":
    from collections import Counter
    lp_loader = BaseLoader(task_type="lp")
    ke_loader = BaseLoader(task_type="ke")
    sk_loader = BaseLoader(task_type="sk")
    dist_rub_lp = Counter([ex["num_rubrics"] for ex in lp_loader.train]) + Counter([ex["num_rubrics"] for ex in lp_loader.val]) + Counter([ex["num_rubrics"] for ex in lp_loader.test_ua]) + Counter([ex["num_rubrics"] for ex in lp_loader.test_uq])
    dist_rub_ke = Counter([ex["num_rubrics"] for ex in ke_loader.train]) + Counter([ex["num_rubrics"] for ex in ke_loader.val]) + Counter([ex["num_rubrics"] for ex in ke_loader.test_ua]) + Counter([ex["num_rubrics"] for ex in ke_loader.test_uq])
    dist_rub_sk = Counter([ex["num_rubrics"] for ex in sk_loader.train]) + Counter([ex["num_rubrics"] for ex in sk_loader.val]) + Counter([ex["num_rubrics"] for ex in sk_loader.test_ua]) + Counter([ex["num_rubrics"] for ex in sk_loader.test_uq])
    print("Learning Performance Rubric Distribution:", dist_rub_lp)
    print("Knowledge Elements Rubric Distribution:", dist_rub_ke)
    print("Skills Rubric Distribution:", dist_rub_sk)
