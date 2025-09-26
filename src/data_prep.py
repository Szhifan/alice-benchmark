from typing import Literal
from datasets import enable_caching, Dataset, disable_caching
import json
import torch
from transformers import AutoTokenizer
import random
"""
Dataprep pipeline: 
1. Load the Alice dataset from json files.
2. Encode the dataset using the provided encoding functions for different model settings.
3. Provide collate functions for batching the dataset.
"""
enable_caching()
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
    # encode answer and sample solution as a sequence pair
    output = tokenizer(example["sample_solution"], example["answer"], max_length=512, truncation=True) 
    for field in output:
        example[field] = output[field]
    example["labels"] = int(example["level"])
    return example
def encode_rubric_pair(example, tokenizer):
    """
    Encode rubric and answer as a sequence pair.
    """
    output = tokenizer(example["answer"], example["rubric"], max_length=512, truncation=True)
    for field in output:
        example[field] = output[field]
    return example
def encode_fields_special_tokens(example, tokenizer, fields: list[str] = ["answer","rubric"]): 
    """
    Encode fields with special tokens.
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
    Encode rubric as one encoding and other fields as another encoding for SNet.
    """
    # Encode rubric separately
    rubric_output = tokenizer(example["rubric"], max_length=512, truncation=True, add_special_tokens=True)
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
    
    return example

def encode_with_fields_snet(
    example, tokenizer, fields: list[str] = ["answer"], 
    add_instruction: bool = False, format: Literal["natural_lang", "structured"] = "structured"
):
    """
    Encoding function for snet llm architecture.
    """
    rubric_encoded = tokenizer(example["rubric"], max_length=512, truncation=True)
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
    rubric_text = [f"Score: {key} Rubric: {value}" for key, value in rubric.items()]
    text2encode = f"Welche der folgenden Rubriken erfüllen die Schülerantwort: {example['answer']}?" + "\n".join(rubric_text) + "\n" \
    + addition_input_text
    if train:
        response = f"Antwort: {example['level']}"
        text2encode += response
    encoded = tokenizer(text2encode, max_length=1024, truncation=True)

    for field in encoded:
        example[field] = encoded[field]
    example["text"] = text2encode
    return example

# collate functions
def gen_collate_fn(input_batch, pad_id=0, return_meta=False):
    """
    basic collate function for batching the input batch.
    Mode: controls whether to return meta information or not.
    """
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids"]) for x in input_batch], batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["attention_mask"]) for x in input_batch], batch_first=True, padding_value=0)
    batch = {
        "input_ids": input_ids,
        "labels": input_ids.clone(), 
        "attention_mask": attention_mask,
    } 
    meta = {
        "id": [x["id"] for x in input_batch],
        "level": [x["level"] for x in input_batch]
    }
    if return_meta:
        return batch, meta
    return batch

def snet_collate_fn(input_batch, pad_id=0, return_meta=False):
    """
    collate function for settings where the rubric and answer are encoded separately.
    """
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids"]) for x in input_batch], batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["attention_mask"]) for x in input_batch], batch_first=True, padding_value=0)

    rubric_input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["rubric_input_ids"]) for x in input_batch], batch_first=True, padding_value=pad_id)
    rubric_attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["rubric_attention_mask"]) for x in input_batch], batch_first=True, padding_value=0)

    batch = {
        "input_ids_a": input_ids,
        "attention_mask_a": attention_mask,
        "input_ids_b": rubric_input_ids,
        "attention_mask_b": rubric_attention_mask,
        "labels": torch.tensor([x["labels"] for x in input_batch]),
    }
    meta = {
        "id": [x["id"] for x in input_batch],
        "rubric_level": [x["rubric_level"] for x in input_batch],
        "level": [x["level"] for x in input_batch],
    }
    if return_meta:
        return batch, meta
    return batch

def xnet_collate_fn(input_batch, pad_id=0, return_meta=False):
    """
    Collate function for grouped dataset where each example contains multiple rubrics
    with input_ids, attention_mask etc. in shape [R, S] format.
    
    Args:
        input_batch: List of examples from grouped dataset
        pad_id: Padding token id
        return_meta: Whether to return metadata
        
    Returns:
        batch: Dict with tensors of shape [B, R, S] where B=batch_size, R=num_rubrics, S=seq_len
        meta: Optional metadata dict
    """
    batch_size = len(input_batch)
    max_rubrics = max(len(x["input_ids"]) for x in input_batch)
    
    # Initialize lists to collect padded sequences
    batch_input_ids = []
    batch_attention_mask = []
    batch_token_type_ids = []
    batch_labels = []
    
    for example in input_batch:
        num_rubrics = len(example["input_ids"])
        
        # Convert input_ids and attention_mask to tensors
        input_ids_tensors = [torch.tensor(seq) for seq in example["input_ids"]]
        attention_mask_tensors = [torch.tensor(seq) for seq in example["attention_mask"]]
        
        # Pad sequences within this example to same length
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids_tensors, batch_first=True, padding_value=pad_id
        )  # [R, S]
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask_tensors, batch_first=True, padding_value=0
        )  # [R, S]
        
        # Handle token_type_ids if present
        if "token_type_ids" in example and example["token_type_ids"] is not None:
            token_type_ids_tensors = [torch.tensor(seq) for seq in example["token_type_ids"]]
            padded_token_type_ids = torch.nn.utils.rnn.pad_sequence(
                token_type_ids_tensors, batch_first=True, padding_value=0
            )
        else:
            padded_token_type_ids = None
        
        # Pad rubrics dimension to max_rubrics if needed
        if num_rubrics < max_rubrics:
            seq_len = padded_input_ids.shape[1]
            # Create padding for missing rubrics
            rubric_padding = torch.full((max_rubrics - num_rubrics, seq_len), pad_id, dtype=torch.long)
            padded_input_ids = torch.cat([padded_input_ids, rubric_padding], dim=0)
            
            mask_padding = torch.zeros((max_rubrics - num_rubrics, seq_len), dtype=torch.long)
            padded_attention_mask = torch.cat([padded_attention_mask, mask_padding], dim=0)
            
            if padded_token_type_ids is not None:
                token_type_padding = torch.zeros((max_rubrics - num_rubrics, seq_len), dtype=torch.long)
                padded_token_type_ids = torch.cat([padded_token_type_ids, token_type_padding], dim=0)
        
        batch_input_ids.append(padded_input_ids)
        batch_attention_mask.append(padded_attention_mask)
        if padded_token_type_ids is not None:
            batch_token_type_ids.append(padded_token_type_ids)
        
        # Use the original labels (not binary)
        batch_labels.append(example["labels"])
    
    # Pad sequence length dimension across batch
    max_seq_len = max(x.shape[1] for x in batch_input_ids)
    
    final_input_ids = []
    final_attention_mask = []
    final_token_type_ids = []
    
    for i in range(batch_size):
        current_seq_len = batch_input_ids[i].shape[1]
        if current_seq_len < max_seq_len:
            # Pad sequence length dimension
            seq_padding = torch.full((max_rubrics, max_seq_len - current_seq_len), pad_id, dtype=torch.long)
            padded_input_ids = torch.cat([batch_input_ids[i], seq_padding], dim=1)
            
            mask_seq_padding = torch.zeros((max_rubrics, max_seq_len - current_seq_len), dtype=torch.long)
            padded_attention_mask = torch.cat([batch_attention_mask[i], mask_seq_padding], dim=1)
            
            if batch_token_type_ids:
                token_seq_padding = torch.zeros((max_rubrics, max_seq_len - current_seq_len), dtype=torch.long)
                padded_token_type_ids = torch.cat([batch_token_type_ids[i], token_seq_padding], dim=1)
        else:
            padded_input_ids = batch_input_ids[i]
            padded_attention_mask = batch_attention_mask[i]
            if batch_token_type_ids:
                padded_token_type_ids = batch_token_type_ids[i]
        
        final_input_ids.append(padded_input_ids)
        final_attention_mask.append(padded_attention_mask)
        if batch_token_type_ids:
            final_token_type_ids.append(padded_token_type_ids)
    
    # Stack to create final batch tensors [B, R, S]
    batch = {
        "input_ids": torch.stack(final_input_ids),  # [B, R, S]
        "attention_mask": torch.stack(final_attention_mask),  # [B, R, S]
        "labels": torch.tensor(batch_labels),  # [B] - original labels
        "num_rubrics": torch.tensor([len(x["input_ids"]) for x in input_batch]),  # [B] - actual number of rubrics per example
    }
    
    if final_token_type_ids:
        batch["token_type_ids"] = torch.stack(final_token_type_ids)  # [B, R, S]
    
    meta = {
        "id": [x["id"] for x in input_batch],
        "level": [x["level"] for x in input_batch],
        "question_id": [x["question_id"] for x in input_batch],
    }
    
    if return_meta:
        return batch, meta
    return batch
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
        }
        
        # Stack the encoded fields
        grouped_example["input_ids"] = [ex["input_ids"] for ex in examples]
        grouped_example["attention_mask"] = [ex["attention_mask"] for ex in examples]
        
        # Handle token_type_ids if present
        if "token_type_ids" in examples[0]:
            grouped_example["token_type_ids"] = [ex["token_type_ids"] for ex in examples]
        
        # Keep labels and rubric levels for each rubric
        grouped_example["labels"] = examples[0]["level"]
        grouped_example["num_rubrics"] = len(examples)
        # Keep other metadata from first example
        for key in examples[0]:
            if key not in ["input_ids", "attention_mask", "token_type_ids", "labels", "num_rubrics"]:
                grouped_example[key] = examples[0][key]
        
        regrouped_examples.append(grouped_example)
    
    return Dataset.from_list(regrouped_examples)
class BaseLoader:
    """
    Load the splits of Alice dataset.
    """
    def __init__(self, train_frac=1, task_type="lp"):
        assert train_frac <= 1 and train_frac > 0, "train_frac must be in (0, 1]"
        assert task_type in ["lp","ke","sk"] , "task_type must be one of ['lp','ke','sk']"
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

        if self.task_type == "lp":
            fields_to_keep = ["id","question_id","question","answer","sample_solution","rubric","level"]
            entry["question"] = meta_info.get("prompt", "")
            entry["sample_solution"] = meta_info.get("solution", "")
            rubric = meta_info["learning_performance"]
            entry["rubric"] = {k: v['rule'] for k, v in rubric.items()}
            entry["level"] = int(next(iter(entry["learning_performance"].values())))
            new_entry = {k: entry[k] for k in entry if k in fields_to_keep}
            return new_entry
        elif self.task_type == "ke":
            expending_entries = []
            if not entry.get("knowledge_elements"):
                return []
            for i, ke in enumerate(entry['knowledge_elements']):
                fields_to_keep = ["id","question_id","question","answer","sample_solution","rubric","knowledge_element","level"]
                new_entry = entry.copy()
                new_entry["question"] = meta_info.get("prompt", "")
                new_entry["sample_solution"] = meta_info.get("solution", "")
                new_entry["knowledge_element"] = ke
                ke_rubric  = meta_info.get("knowledge_elements", {}).get(ke, {})
                ke_rubric = {k: f"{ke}: {v['description']}" for k, v in ke_rubric.items()}
                new_entry["rubric"] = ke_rubric
                new_entry["knowledge_element"] = ke
                new_entry["level"] = int(entry["knowledge_elements"][ke])
                new_entry["id"] = f"{entry['id']}_ke{i}"
                new_entry = {k: new_entry[k] for k in new_entry if k in fields_to_keep}
                expending_entries.append(new_entry)
            return expending_entries
        elif self.task_type == "sk":
            if not entry.get("skills"):
                return []
            expending_entries = []
            for i, sk in enumerate(entry['skills']):
                fields_to_keep = ["id","question_id","question","answer","sample_solution","rubric","skill_element","level"]
                new_entry = entry.copy()
                new_entry["question"] = meta_info.get("prompt", "")
                new_entry["sample_solution"] = meta_info.get("solution", "")
                new_entry["skills"] = sk
                sk_rubric  = meta_info.get("skills", {}).get(sk, {})
                sk_rubric = {k: f"{sk}: {v['description']}" for k, v in sk_rubric.items()}
                new_entry["rubric"] = sk_rubric
                new_entry["skills"] = sk
                new_entry["level"] = int(entry["skills"][sk])
                new_entry["id"] = f"{entry['id']}_sk{i}"
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
                for level, rb in rubric.items():
                    new_example = example.copy()
                    new_example["rubric"] = rb
                    new_example["rubric_level"] = int(level)
                    new_example["labels"] = 1 if int(new_example["level"]) == int(level) else 0
                    expanded_data.append(new_example)
            expanded_data = Dataset.from_list(expanded_data)
            return expanded_data
        self.train = _expand_dataset(self.train)
        self.val = _expand_dataset(self.val)
        self.test_ua = _expand_dataset(self.test_ua)
        self.test_uq = _expand_dataset(self.test_uq)



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    loader = RubricRetrievalLoader(train_frac=1, task_type="lp")
    loader.test_uq = encode_dataset(loader.test_uq, AutoTokenizer.from_pretrained("bert-base-multilingual-cased"), encode_rubric_pair)
    loader.test_uq = group_by_id(loader.test_uq)
    dataloader = DataLoader(loader.test_uq, batch_size=2, collate_fn=xnet_collate_fn)   
    for batch in dataloader:
        print(batch)
        break