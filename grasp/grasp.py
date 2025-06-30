from transformers import ElectraPreTrainedModel, ElectraModel, AutoTokenizer, AutoConfig, PretrainedConfig, BertPreTrainedModel, BertModel
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from transformers import get_scheduler

import pandas as pd
import json

from datasets import Dataset

def read_dataset(path):
    df = pd.read_excel(path)

    outp = []
    for i in range(len(df)):
        ruj = json.loads(df['rubric'].values[i])

        datapoint = {
            'question': str(df['question'].values[i]),
            'answer': str(df['answer'].values[i]),
            'rubrics': [
                str(ruj[s]) for s in ruj
            ],
            'label': int(df['level'].values[i])
        }
        outp.append(datapoint)

    return Dataset.from_list(outp)

class PointerCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        input_keys = ["input_ids", "attention_mask", "labels"]
        batch = {k: torch.tensor([f[k] for f in features]) for k in input_keys}

        # Pad rubric spans
        max_rubrics = max(len(f["rubrik_spans"]) for f in features)
        span_tensor = torch.zeros(len(features), max_rubrics, 2, dtype=torch.long)
        mask_tensor = torch.zeros(len(features), max_rubrics, dtype=torch.bool)
        for i, f in enumerate(features):
            rs = torch.tensor(f["rubrik_spans"], dtype=torch.long)
            span_tensor[i, :rs.shape[0], :] = rs
            mask_tensor[i, :rs.shape[0]] = 1
        batch["rubrik_spans"] = span_tensor
        batch["rubrik_mask"] = mask_tensor

        # Answer spans
        batch["answer_span"] = torch.tensor([f["answer_span"] for f in features], dtype=torch.long)

        return batch

class PointerRubricConfig(PretrainedConfig):
    model_type = "pointer-rubric"

    def __init__(self, hidden_size=768, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size

class PointerRubricModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.pointer_head = nn.Bilinear(config.hidden_size, config.hidden_size, 1)

        self.init_weights()

    def forward(
        self, 
        input_ids, 
        attention_mask, 
        rubrik_spans, 
        answer_span, 
        rubrik_mask, 
        labels=None
    ):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # [B, T, H]

        B, R = rubrik_spans.shape[:2]
        H = token_embeddings.shape[-1]

        rubrik_embeddings = []
        for i in range(R):
            starts = rubrik_spans[:, i, 0]
            ends = rubrik_spans[:, i, 1]
            emb = torch.stack([
                token_embeddings[b, starts[b]:ends[b]].mean(dim=0)
                if rubrik_mask[b, i]
                else torch.zeros(H, device=token_embeddings.device)
                for b in range(B)
            ])
            rubrik_embeddings.append(emb)
        rubrik_embeddings = torch.stack(rubrik_embeddings, dim=1)  # [B, R, H]

        a_starts = answer_span[:, 0]
        a_ends = answer_span[:, 1]
        answer_emb = torch.stack([
            token_embeddings[b, a_starts[b]:a_ends[b]].mean(dim=0)
            for b in range(B)
        ])  # [B, H]

        answer_exp = answer_emb.unsqueeze(1).expand(-1, R, -1)
        scores = self.pointer_head(rubrik_embeddings, answer_exp).squeeze(-1)  # [B, R]

        scores = scores.masked_fill(~rubrik_mask, float("-inf"))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(scores, labels)
        return {"logits": scores, "loss": loss}

tokenizer = AutoTokenizer.from_pretrained("deepset/gbert-large")
collator = PointerCollator(tokenizer)

def tokenize_with_spans(example):
    rubriken = example["rubrics"]
    question = example["question"]
    answer = example["answer"]

    tokens = ["[CLS]"]
    span_indices = []

    for rubrik in rubriken:
        start = len(tokens)
        rubrik_tokens = tokenizer.tokenize(rubrik)
        tokens.extend(rubrik_tokens + ["[SEP]"])
        end = len(tokens)
        span_indices.append((start, end))

    answer_start = len(tokens)
    qa_tokens = tokenizer.tokenize("Question: " + question + " Answer: " + answer)
    tokens.extend(qa_tokens)
    answer_end = len(tokens)

    enc = tokenizer(tokens, is_split_into_words=True, truncation=True, padding="max_length")

    enc["rubrik_spans"] = span_indices
    enc["answer_span"] = [answer_start, answer_end]
    enc["labels"] = example["label"]

    return enc

dataset = read_dataset("ALICE_train_new.xlsx")  # needs to contain rubrics, question, answer, label
tokenized_dataset = dataset.map(tokenize_with_spans)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./pointer-rubric",
    per_device_train_batch_size=1,  # Lower from 4 to 2
    gradient_accumulation_steps=4,  # Add this for similar effective batch size
    learning_rate=5e-6,
    max_grad_norm=1.0,
    optim='adamw_torch',
    num_train_epochs=6,
    save_strategy="no",
    logging_steps=50,
    lr_scheduler_type="linear",
    warmup_steps=400
)

conf = PointerRubricConfig.from_pretrained('deepset/gelectra-large')
conf._name_or_path = 'deepset/gelectra-large'  # Save base encoder name
model = PointerRubricModel(conf)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collator
)

trainer.train()

model.save_pretrained("./grasp-1")
tokenizer.save_pretrained("./grasp-1")

