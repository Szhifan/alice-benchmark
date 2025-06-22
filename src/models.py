import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, T5Config
import torch
from transformers import AutoModel,T5EncoderModel
from torch.nn import CrossEntropyLoss, Bilinear
from torch.nn.functional import sigmoid
from dataclasses import dataclass
from typing import Optional, Tuple
import re 
@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
def mean_pooling(
    model_output: ModelOutput, 
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Perform mean pooling on the model output.
    """
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask 
def get_tokenizer(model_name: str) -> AutoTokenizer:

    return AutoTokenizer.from_pretrained(model_name)

def freeze_bert_layers(model, n_frozen_layers: int):
    """
    Freeze the specified number of layers in the BERT model.
    """
    for name, param in model.named_parameters():
        regex = re.compile(r"layer\.(\d+)")
        match = regex.search(name)
        if match:
            layer_num = int(match.group(1))
            if layer_num < n_frozen_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
def freeze_bert_embeddings(model):
    """
    Freeze the embedding layer of the BERT model.
    """
    for param in model.embeddings.parameters():
        param.requires_grad = False
def freeze_t5_layers(model, n_frozen_layers: int):
    """
    Freeze the specified number of layers in the T5 model.
    """
    for name, param in model.named_parameters():
        regex = re.compile(r"block\.(\d+)")
        match = regex.search(name)
        if match:
            layer_num = int(match.group(1))
            if layer_num < n_frozen_layers:
                param.requires_grad = False
            else:
                param.requires_grad = True
def freeze_t5_embeddings(model):
    """
    Freeze the embedding layer of the T5 model.
    """
    for param in model.shared.parameters():
        param.requires_grad = False
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
class AsagCrossEncoder(nn.Module):
    """
    Encoder based ASAG model
    """
    def __init__(
        self, 
        model_name: str, 
        num_labels: int = 2, 
        freeze_layers: int = 0, 
        freeze_embeddings: bool = False, 
        label_weights: Optional[torch.Tensor] = None, 
        use_ce_loss: bool = True
    ):
        super().__init__()
        self.is_t5 = "t5" in model_name
        self.model_name = model_name 
        self.encoder = AutoModel.from_pretrained(model_name) if not self.is_t5 else T5EncoderModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.label_weights = label_weights
        self.num_labels = num_labels if use_ce_loss else 1
        self.classifier = ClassificationHead(hidden_size, self.num_labels)
        
        self.use_ce_loss = use_ce_loss
        if self.is_t5:
            freeze_t5_layers(self.encoder, freeze_layers)
        else:
            freeze_bert_layers(self.encoder, freeze_layers)
        if freeze_embeddings:
            if self.is_t5:
                freeze_t5_embeddings(self.encoder)
            else:
                freeze_bert_embeddings(self.encoder)
    def get_loss_ce(self, logits: torch.Tensor, label_id: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-entropy loss.
        """
        self.label_weights = self.label_weights.to(logits.device) if self.label_weights is not None else None
        loss_fct = CrossEntropyLoss(weight=self.label_weights)
        return loss_fct(logits.view(-1, self.num_labels), label_id.view(-1))
    def get_loss_mse(self, logits: torch.Tensor, label_id: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error loss.
        """
        loss_fct = nn.MSELoss()
        if logits.device == label_id.device == "mps":
            logits = logits.float()
            label_id = label_id.float()
        return loss_fct(logits.view(-1, self.num_labels), label_id.view(-1, self.num_labels))
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        token_type_ids: Optional[torch.Tensor] = None, 
        label_id: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        if self.is_t5:
            logits = self.forward_t5(input_ids, attention_mask, label_id)
        else:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            sequence_output = encoder_outputs.last_hidden_state
            logits = self.classifier(sequence_output[:, 0, :])  # Use [CLS] token representation


        loss = None
        if label_id is not None:
            loss = self.get_loss_ce(logits, label_id) if self.use_ce_loss else self.get_loss_mse(logits, label_id)

        return ModelOutput(logits=logits, loss=loss)
    def forward_t5(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_id: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        """
        Forward method for T5 model. Extracts the last EOS token as the sequence representation.
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        sequence_output = encoder_outputs.last_hidden_state

        # Extract the position of </s> token (id = tokenizer.eos_token_id)
        eos_token_id = self.encoder.config.eos_token_id
        eos_mask = input_ids.eq(eos_token_id)
        batch_size, hidden_size = sequence_output.size(0), sequence_output.size(-1)
        sequence_output = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]

        logits = self.classifier(sequence_output)
        return logits
class PointerRubricModel(nn.Module):
    """
    Pointer Rubric Model for ASAG.
    This model uses a T5 encoder to process the input and a pointer mechanism to select the rubric.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name) if "t5" not in model_name else T5EncoderModel.from_pretrained(model_name)
        self.pointer_head = nn.Bilinear(self.encoder.config.hidden_size, self.encoder.config.hidden_size, 1)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rubric_span: torch.Tensor,
        answer_span: torch.Tensor,
        rubric_mask: torch.Tensor,
        label_id: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        """
        Forward method for the Pointer Rubric Model.
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        seq_embeddings = encoder_outputs.last_hidden_state
        B, R = rubric_span.size(0), rubric_span.size(1)
        H = seq_embeddings.size(-1)
        rubric_embeddings = torch.zeros((B, R, H), dtype=seq_embeddings.dtype, device=seq_embeddings.device)
        for i in range(R):
            starts, ends = rubric_span[:, i, 0], rubric_span[:, i, 1]
            emb = torch.stack([
                seq_embeddings[b, starts[b]:ends[b]].mean(dim=0) 
                if rubric_mask[b, i] else torch.zeros(H, dtype=seq_embeddings.dtype, device=seq_embeddings.device)
                for b in range(B)
            ])
            rubric_embeddings[:, i, :] = emb
        ans_starts, ans_ends = answer_span[:, 0], answer_span[:, 1]
        answer_embeddings = torch.stack([
            seq_embeddings[b, ans_starts[b]:ans_ends[b]].mean(dim=0) 
            for b in range(B)
        ])
        answer_embeddings = answer_embeddings.unsqueeze(1).expand(-1, R, -1)  # Expand to match rubric embeddings
        scores = self.pointer_head(rubric_embeddings, answer_embeddings).squeeze(-1)  # Shape: (B, R)
        scores = scores.masked_fill(~rubric_mask, float('-inf'))  # Apply mask to scores
        logits = F.softmax(scores, dim=-1)  # Convert scores to probabilities
        loss = None
        if label_id is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), label_id.view(-1))
        return ModelOutput(logits=logits, loss=loss)
if __name__ == "__main__":
    # Import T5 tokenizer

    from data_prep_asap import AsapRubricPointer, encoding_with_rubric_span
    from torch.utils.data import DataLoader
    dts = AsapRubricPointer()
    tokenizer = get_tokenizer("bert-base-uncased")
    test_ds = dts.test
    test_ds = test_ds.map(lambda x: encoding_with_rubric_span(x, tokenizer))
    
    loader = DataLoader(
        test_ds, 
        batch_size=16, 
        collate_fn=dts.collate_fn,
        shuffle=True,
    )
    for batch, meta in loader:
        model = PointerRubricModel("bert-base-uncased")
        output = model(**batch)
        print(output.logits)
        break