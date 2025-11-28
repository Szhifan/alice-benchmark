import torch
from typing import Literal
FIELD_EN2DE = {
    "answer": "Antwort",
    "rubric": "Rubrik",
    "question": "Frage",
    "sample_solution": "Lösung"
}
## encoding functions 
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

def encode_sequence_bert(example, tokenizer, fields: list[str] = ["answer","rubric"]): 
    """
    For bert-like models.
    Encode multiple fields with special tokens (SEP).
    """
    def default_encode(example, tokenizer):
        """
        For bert-like models.
        Encode rubric and answer as a sequence pair.
        """
        output = tokenizer(example["answer"], example["rubric"], max_length=512, truncation=True)
        for field in output:
            example[field] = output[field]
        return example
    if set(fields) == {"answer", "rubric"}:
        return default_encode(example, tokenizer)
    text2encode = []
    for field in fields: 
        if field not in example:
            raise ValueError(f"Field '{field}' not found in the example.")
        text2encode.append(example[field])

    text2encode = tokenizer.sep_token.join(text2encode)
    output = tokenizer(text2encode, max_length=512, truncation=True, add_special_tokens=True)
    return output
def encode_sequence_llm(example, tokenizer, fields: list[str] = ["answer","rubric"], add_instruction: bool = False, format: Literal["natural_lang", "structured"] = "natural_lang"):
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

## collate functions 
def base_collate_fn(input_batch, pad_id=0, return_meta=False):
    """
    Basic collate function for batching the input batch.
    Mode: controls whether to return meta information or not.
    """
    input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["input_ids"]) for x in input_batch], batch_first=True, padding_value=pad_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence([torch.tensor(x["attention_mask"]) for x in input_batch], batch_first=True, padding_value=0)
    labels = torch.tensor([x["labels"] for x in input_batch])
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    } 
    meta = {
        "id": [x["id"] for x in input_batch],
        "level": [x["level"] for x in input_batch],
        "rubric_level": [x.get("rubric_level", None) for x in input_batch],
    }
    if return_meta:
        return batch, meta
    return batch


def xnet_collate_fn(input_batch, pad_id=0, return_meta=False, mask_prob=0.0):
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
    max_rubrics = max([x["num_rubrics"] for x in input_batch])
    # Initialize lists to collect padded sequences
    batch_input_ids = []
    batch_attention_mask = []
    batch_token_type_ids = []
    batch_labels = []
    
    for example in input_batch:
        num_rubrics = example["num_rubrics"]
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
        "num_rubrics": torch.tensor([x["num_rubrics"] for x in input_batch]),  # [B] - actual number of rubrics per example,
        "mask_prob": mask_prob
    }
    
    if final_token_type_ids:
        batch["token_type_ids"] = torch.stack(final_token_type_ids)  # [B, R, S]
    
    meta = {
        "id": [x["id"] for x in input_batch],
        "level": [x["level"] for x in input_batch],
        "question_id": [x["question_id"] for x in input_batch],
        "rubric_level": [x["rubric_level"] for x in input_batch],
    }
    
    if return_meta:
        return batch, meta
    return batch