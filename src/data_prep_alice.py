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
        text2encode = "Determine if rubric is satisfied by the answer:\n" + text2encode
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
def encode_with_fields_separate_rubric(
    example, tokenizer, fields: list[str] = ["answer"], 
    add_instruction: bool = False, format: Literal["natural_lang", "structured"] = "natural_lang"
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
            query2encode += f"{field}: {example[field]}\n"
        elif format == "structured":
            query2encode += f"<{field}>{example[field]}</{field}>\n"
    if add_instruction:
        query2encode = "Determine if rubric is satisfied by the answer:\n" + query2encode
    query_output = tokenizer(query2encode, max_length=512, truncation=True)
    for field in query_output:
        example[field] = query_output[field]
    for field in rubric_encoded:
        example[f"rubric_{field}"] = rubric_encoded[field]
    return example