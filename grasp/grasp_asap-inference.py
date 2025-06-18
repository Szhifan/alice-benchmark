from transformers import PreTrainedModel, AutoModel, AutoTokenizer, ModernBertConfig, PreTrainedModel, AutoConfig
from transformers import Trainer, TrainingArguments
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import DataLoader

from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix

from transformers import get_scheduler

from safetensors.torch import load_file

from tqdm import tqdm

import pandas as pd
import json

from datasets import Dataset

torch.set_float32_matmul_precision('high')

def read_dataset(df):
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

class PointerRubricModel(PreTrainedModel):
    config_class = ModernBertConfig
    def __init__(self, config):
        super().__init__(config)
        # Dynamically load the encoder model based on the config.
        self.encoder = AutoModel.from_config(config)
        self.dropout = nn.Dropout(0.1)
        self.pointer_head = nn.Bilinear(config.hidden_size, config.hidden_size, 1)

    def load_encoder(self, model_name_or_path):
        # Load a model from a given path or model name
        self.encoder = AutoModel.from_pretrained(model_name_or_path)

    def forward(self, input_ids, attention_mask, rubrik_spans, answer_span, rubrik_mask, labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # [B, T, H]

        B, R = rubrik_spans.shape[:2]
        H = token_embeddings.shape[-1]

        # Process rubric spans
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

        # Process answer spans
        a_starts = answer_span[:, 0]
        a_ends = answer_span[:, 1]
        answer_emb = torch.stack([
            token_embeddings[b, a_starts[b]:a_ends[b]].mean(dim=0)
            for b in range(B)
        ])  # [B, H]

        # Compute the pointer scores
        answer_exp = answer_emb.unsqueeze(1).expand(-1, R, -1)
        scores = self.pointer_head(rubrik_embeddings, answer_exp).squeeze(-1)  # [B, R]

        # Mask the scores where rubrik_mask is False
        scores = scores.masked_fill(~rubrik_mask, float("-inf"))

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(scores, labels)
        return {"logits": scores, "loss": loss}

    # Ensure the custom layers (head) are part of the model's weights
    def save_pretrained(self, save_directory):
        super().save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # Load the model with the specified pre-trained path or name
        model = super().from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
        return model

tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
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

    enc = tokenizer(tokens, is_split_into_words=True, truncation=True, padding="longest")

    enc["rubrik_spans"] = span_indices
    enc["answer_span"] = [answer_start, answer_end]
    enc["labels"] = example["label"]

    return enc

# 1. Load config manually
config = ModernBertConfig.from_pretrained("./grasp-asap-0")

# 2. Initialize your custom model
model = PointerRubricModel(config)

# 3. Load weights manually
state_dict = load_file("./grasp-asap-0/model.safetensors")
model.load_state_dict(state_dict)

# 4. Send to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

df = pd.read_excel(f"ASAP_test.xlsx")
for task in range(1, 11, 1):
    sub_df = df[df['question_id'] == task]
    print(sub_df)
    dataset = read_dataset(sub_df)  # needs to contain rubrics, question, answer, label

    tokenized_dataset = dataset.map(tokenize_with_spans)
    dataloader = DataLoader(tokenized_dataset, batch_size=1, collate_fn=collator)

    pred = []
    ref = [int(l) for l in sub_df['level'].values]
    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)
            pred.extend(preds.cpu().tolist())

    print(dataset)
    print(f1_score(ref, pred, labels=range(0, sub_df['maxlevel'].values[0] + 1), average='weighted'))
    print(cohen_kappa_score(ref, pred, labels=range(0, sub_df['maxlevel'].values[0] + 1), weights='quadratic'))
    print(confusion_matrix(ref, pred, labels=range(0, sub_df['maxlevel'].values[0] + 1)))   
    