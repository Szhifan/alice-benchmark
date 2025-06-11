import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, T5Config

import torch
from transformers import AutoModel,T5EncoderModel,T5ForConditionalGeneration
from torch.nn import CrossEntropyLoss
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
class Asag_CrossEncoder(nn.Module):
    """
    Encoder based ASAG model
    """
    def __init__(self, model_name: str, num_labels: int, freeze_layers: int = 0, freeze_embeddings: bool = False):
    
        super().__init__()
        self.is_t5 = "t5" in model_name
        self.model_name = model_name 
        self.encoder = AutoModel.from_pretrained(model_name) if not self.is_t5 else T5EncoderModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.classifier = ClassificationHead(hidden_size, num_labels)
        self.num_labels = num_labels
        if freeze_layers > 0:
            if self.is_t5:
                freeze_t5_layers(self.encoder, freeze_layers)
            else:
                freeze_bert_layers(self.encoder, freeze_layers)
        if freeze_embeddings:
            if self.is_t5:
                freeze_t5_embeddings(self.encoder)
            else:
                freeze_bert_embeddings(self.encoder)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        token_type_ids: Optional[torch.Tensor] = None, 
        label_id: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        if self.is_t5:
            return self.forward_t5(input_ids, attention_mask, label_id)
 
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = encoder_outputs.last_hidden_state
        logits = self.classifier(sequence_output[:, 0, :])  # Use [CLS] token representation


        loss = None
        if label_id is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label_id.view(-1))

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

        loss = None
        if  label_id is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), label_id.view(-1))

        return ModelOutput(logits=logits, loss=loss)
   

  
    
if __name__ == "__main__":
    # Import T5 tokenizer
    from transformers import T5Tokenizer
    # Define model name and tokenizer
    model = Asag_CrossEncoder("bert-base-uncased", 6, freeze_layers=10, freeze_embeddings=True)
    for name, param in model.named_parameters():
        print(name, param.requires_grad)