import torch
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
    }
    if return_meta:
        return batch, meta
    return batch

def snet_collate_fn(input_batch, pad_id=0, return_meta=False, mask_prob=0.0):
    """
    Collate function for SNet batch format where answer and rubric are encoded separately.
    Expects grouped dataset where each example contains one answer and multiple rubrics.
    
    Args:
        input_batch: List of examples from grouped dataset
        pad_id: Padding token id
        return_meta: Whether to return metadata
        
    Returns:
        batch: Dict with tensors - answer [B, S_answer], rubric [B, R, S_rubric]
        meta: Optional metadata dict
    """
    batch_size = len(input_batch)
    max_rubrics = max(len(x["rubric_input_ids"]) for x in input_batch)
    
    # Initialize lists for answer and rubric sequences
    batch_answer_input_ids = []
    batch_answer_attention_mask = []
    batch_answer_token_type_ids = []
    
    batch_rubric_input_ids = []
    batch_rubric_attention_mask = []
    batch_rubric_token_type_ids = []
    
    batch_labels = []
    
    for example in input_batch:
        num_rubrics = len(example["rubric_input_ids"])
        
        answer_input_ids = torch.tensor(example["input_ids"])
        answer_attention_mask = torch.tensor(example["attention_mask"])
        
        # Process rubric sequences
        rubric_input_ids_tensors = [torch.tensor(seq) for seq in example["rubric_input_ids"]]
        rubric_attention_mask_tensors = [torch.tensor(seq) for seq in example["rubric_attention_mask"]]
        
        # Pad rubric sequences within this example
        padded_rubric_input_ids = torch.nn.utils.rnn.pad_sequence(
            rubric_input_ids_tensors, batch_first=True, padding_value=pad_id
        )  # [R, S]
        padded_rubric_attention_mask = torch.nn.utils.rnn.pad_sequence(
            rubric_attention_mask_tensors, batch_first=True, padding_value=0
        )  # [R, S]
        
        # Handle token_type_ids if present
        answer_token_type_ids = None
        padded_rubric_token_type_ids = None
        
        if "token_type_ids" in example and example["token_type_ids"] is not None:
            answer_token_type_ids = torch.tensor(example["token_type_ids"])
        
        if "rubric_token_type_ids" in example and example["rubric_token_type_ids"] is not None:
            rubric_token_type_ids_tensors = [torch.tensor(seq) for seq in example["rubric_token_type_ids"]]
            padded_rubric_token_type_ids = torch.nn.utils.rnn.pad_sequence(
                rubric_token_type_ids_tensors, batch_first=True, padding_value=0
            )
        
        # Pad rubrics dimension to max_rubrics if needed
        if num_rubrics < max_rubrics:
            rubric_seq_len = padded_rubric_input_ids.shape[1]
            
            # Create padding for missing rubrics
            rubric_rubric_padding = torch.full((max_rubrics - num_rubrics, rubric_seq_len), pad_id, dtype=torch.long)
            padded_rubric_input_ids = torch.cat([padded_rubric_input_ids, rubric_rubric_padding], dim=0)
            
            rubric_mask_padding = torch.zeros((max_rubrics - num_rubrics, rubric_seq_len), dtype=torch.long)
            padded_rubric_attention_mask = torch.cat([padded_rubric_attention_mask, rubric_mask_padding], dim=0)
            
            # Handle token_type_ids padding
            if padded_rubric_token_type_ids is not None:
                rubric_token_type_padding = torch.zeros((max_rubrics - num_rubrics, rubric_seq_len), dtype=torch.long)
                padded_rubric_token_type_ids = torch.cat([padded_rubric_token_type_ids, rubric_token_type_padding], dim=0)
        
        batch_answer_input_ids.append(answer_input_ids)
        batch_answer_attention_mask.append(answer_attention_mask)
        
        batch_rubric_input_ids.append(padded_rubric_input_ids)
        batch_rubric_attention_mask.append(padded_rubric_attention_mask)
        
        if answer_token_type_ids is not None:
            batch_answer_token_type_ids.append(answer_token_type_ids)
        if padded_rubric_token_type_ids is not None:
            batch_rubric_token_type_ids.append(padded_rubric_token_type_ids)
        
        batch_labels.append(example["labels"])
    
    # Pad answer sequences across batch
    padded_answer_input_ids = torch.nn.utils.rnn.pad_sequence(
        batch_answer_input_ids, batch_first=True, padding_value=pad_id
    )  # [B, S_answer]
    padded_answer_attention_mask = torch.nn.utils.rnn.pad_sequence(
        batch_answer_attention_mask, batch_first=True, padding_value=0
    )  # [B, S_answer]
    
    # Pad rubric sequence length dimension across batch
    max_rubric_seq_len = max(x.shape[1] for x in batch_rubric_input_ids)
    
    final_rubric_input_ids = []
    final_rubric_attention_mask = []
    final_rubric_token_type_ids = []
    
    for i in range(batch_size):
        # Handle rubric sequences
        current_rubric_seq_len = batch_rubric_input_ids[i].shape[1]
        if current_rubric_seq_len < max_rubric_seq_len:
            # Pad sequence length dimension for rubrics
            rubric_seq_padding = torch.full((max_rubrics, max_rubric_seq_len - current_rubric_seq_len), pad_id, dtype=torch.long)
            padded_rubric_input_ids = torch.cat([batch_rubric_input_ids[i], rubric_seq_padding], dim=1)
            
            rubric_mask_seq_padding = torch.zeros((max_rubrics, max_rubric_seq_len - current_rubric_seq_len), dtype=torch.long)
            padded_rubric_attention_mask = torch.cat([batch_rubric_attention_mask[i], rubric_mask_seq_padding], dim=1)
            
            if batch_rubric_token_type_ids:
                rubric_token_seq_padding = torch.zeros((max_rubrics, max_rubric_seq_len - current_rubric_seq_len), dtype=torch.long)
                padded_rubric_token_type_ids = torch.cat([batch_rubric_token_type_ids[i], rubric_token_seq_padding], dim=1)
        else:
            padded_rubric_input_ids = batch_rubric_input_ids[i]
            padded_rubric_attention_mask = batch_rubric_attention_mask[i]
            if batch_rubric_token_type_ids:
                padded_rubric_token_type_ids = batch_rubric_token_type_ids[i]
        
        final_rubric_input_ids.append(padded_rubric_input_ids)
        final_rubric_attention_mask.append(padded_rubric_attention_mask)
        
        if batch_rubric_token_type_ids:
            final_rubric_token_type_ids.append(padded_rubric_token_type_ids)
    
    # Stack to create final batch tensors
    batch = {
        "input_ids_a": padded_answer_input_ids,  # [B, S_answer]
        "attention_mask_a": padded_answer_attention_mask,  # [B, S_answer]
        "input_ids_b": torch.stack(final_rubric_input_ids),  # [B, R, S_rubric]
        "attention_mask_b": torch.stack(final_rubric_attention_mask),  # [B, R, S_rubric]
        "labels": torch.tensor(batch_labels),  # [B]
        "num_rubrics": torch.tensor([len(x["rubric_input_ids"]) for x in input_batch]),  # [B]
        "mask_prob": mask_prob
    }
    
    if batch_answer_token_type_ids:
        padded_answer_token_type_ids = torch.nn.utils.rnn.pad_sequence(
            batch_answer_token_type_ids, batch_first=True, padding_value=0
        )
        batch["token_type_ids_a"] = padded_answer_token_type_ids  # [B, S_answer]
    
    if final_rubric_token_type_ids:
        batch["token_type_ids_b"] = torch.stack(final_rubric_token_type_ids)  # [B, R, S_rubric]
    
    meta = {
        "id": [x["id"] for x in input_batch],
        "level": [x["level"] for x in input_batch],
        "question_id": [x["question_id"] for x in input_batch],
    }
    
    if return_meta:
        return batch, meta
    return batch



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
        "level": [x["level"] for x in input_batch],
        "text": [x["text"] for x in input_batch],
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
    }
    
    if return_meta:
        return batch, meta
    return batch