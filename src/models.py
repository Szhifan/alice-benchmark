import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import torch
from transformers import AutoModel, T5EncoderModel
from torch.nn import CrossEntropyLoss, Bilinear, CosineEmbeddingLoss
from torch.nn.functional import sigmoid, cosine_similarity
from dataclasses import dataclass
from typing import Optional, Tuple
import re 
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    bias="none"
)
@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
def mean_pooling(
    token_embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Perform mean pooling on the model output.
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask 
def eos_embeddings(hiden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Get the embeddings of the last non-padding token in each sequence.
    """
    eos_pos = attention_mask.sum(dim=1) - 1  # End of sequence positions
    return hiden_states[torch.arange(hiden_states.size(0)), eos_pos]
def get_tokenizer(base_model: str) -> AutoTokenizer:

    return AutoTokenizer.from_pretrained(base_model)
def freeze_model(model: nn.Module):
    """
    Freeze all parameters in the model.
    """
    for param in model.parameters():
        param.requires_grad = False
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
    encoder.encoder.block.0.layer.0.SelfAttention.q.weight
    """
    for name, param in model.named_parameters():
        regex = re.compile(r"block\.(\d+)\.layer")
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

    def __init__(self, hidden_size: int, n_labels: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.out_proj = nn.Linear(hidden_size, n_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
class AsagXNet(nn.Module):
    """
    Encoder based ASAG model
    """
    def __init__(
        self, 
        base_model: str, 
        n_labels: int = 2, 
        freeze_layers: int = 0, 
        freeze_embeddings: bool = False, 
        label_weights: Optional[torch.Tensor] = None, 
        use_lora: bool = False,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        if use_lora:
            # freeze_model(self.encoder)  # Freeze all parameters if using LoRA
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()
            
        hidden_size = self.encoder.config.hidden_size
        self.label_weights = label_weights
        self.n_labels = n_labels 
        self.classifier = ClassificationHead(hidden_size, self.n_labels)
        if freeze_layers > 0 and not use_lora:
            freeze_bert_layers(self.encoder, freeze_layers)
        if freeze_embeddings and not use_lora:
            freeze_bert_embeddings(self.encoder)
    def get_loss_ce(self, logits: torch.Tensor, label_id: torch.Tensor) -> torch.Tensor:
   
        self.label_weights = self.label_weights.to(logits.device) if self.label_weights is not None else None
        loss_fct = CrossEntropyLoss(weight=self.label_weights)
        return loss_fct(logits.view(-1, self.n_labels), label_id.view(-1)), logits
    def get_loss_mse(self, logits: torch.Tensor, label_id: torch.Tensor) -> torch.Tensor:
 
        loss_fct = nn.MSELoss()
        prob =  sigmoid(logits)
        label_id = label_id.float()
        return loss_fct(prob.view(-1, self.n_labels), label_id.view(-1, self.n_labels)), prob
    def get_loss(self, logits: torch.Tensor, label_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss based on the number of labels.
        """
        if self.n_labels > 1:
            """
            Model outputs as binary classification logits. 0: rubric is not relevant, 1: rubric is relevant.
            During inference, use the logits of label 1 to determine relevance.
            """
            return self.get_loss_ce(logits, label_id)
        else:
            return self.get_loss_mse(logits, label_id)
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        token_type_ids: Optional[torch.Tensor] = None, 
        label_id: Optional[torch.Tensor] = None
    ) -> ModelOutput:
  
        
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        sequence_output = encoder_outputs.last_hidden_state
        logits = self.classifier(sequence_output[:, 0, :])  # Use [CLS] token representation

        loss = None
        if label_id is not None:
            # Cross-entropy: the model detemine if the rubric is relevant or not
            # MSE: the model predicts the relevance score (0-1) for each rubric
            loss, logits = self.get_loss(logits, label_id)

        return ModelOutput(logits=logits, loss=loss)
class AsagXNetT5(AsagXNet):
    """
    T5-based Encoder for ASAG model
    """

    def __init__(
        self, 
        base_model: str, 
        n_labels: int = 2, 
        freeze_layers: int = 0, 
        freeze_embeddings: bool = False, 
        label_weights: Optional[torch.Tensor] = None, 
        use_lora: bool = False
    ):
        nn.Module.__init__(self)
        self.encoder = T5EncoderModel.from_pretrained(base_model)
        if use_lora:
            # freeze_model(self.encoder)
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()
        hidden_size = self.encoder.config.hidden_size
        self.label_weights = label_weights
        self.n_labels = n_labels
        self.classifier = ClassificationHead(hidden_size, self.n_labels)
        if freeze_layers > 0 and not use_lora:
            freeze_t5_layers(self.encoder, freeze_layers)
        if freeze_embeddings and not use_lora:
            freeze_t5_embeddings(self.encoder)
            
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        label_id: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        # T5 doesn't use token_type_ids
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # T5 outputs the last hidden state differently than BERT-based models

        seq_emb = eos_embeddings(outputs.last_hidden_state, attention_mask)  # Get the last </s> token embeddings
        # Use the first token representation for classification
        logits = self.classifier(seq_emb)

        loss = None
        if label_id is not None:
            loss, logits = self.get_loss(logits, label_id)
        return ModelOutput(logits=logits, loss=loss)
class PointerRubricModel(nn.Module):
    """
    grasp
    """
    def __init__(self, base_model: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)

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

class PointerRubricBiEncoder(nn.Module):
    """
    Grasp, but rubrics and qa are encoded separately.
    """
    def __init__(self, base_model: str, share_encoders: bool = True):
        super().__init__()
        if share_encoders:
            self.ans_encoder = self.rubric_encoder = AutoModel.from_pretrained(base_model)
        else:
            self.ans_encoder = AutoModel.from_pretrained(base_model)
            self.rubric_encoder = AutoModel.from_pretrained(base_model)
        hidden_size = self.ans_encoder.config.hidden_size
        self.pointer_head = nn.Bilinear(hidden_size, hidden_size, 1)
        
        
    def forward(
        self,
        ans_input_ids: torch.Tensor,
        ans_attention_mask: torch.Tensor,
        rubric_input_ids: torch.Tensor,
        rubric_attention_mask: torch.Tensor,
        rubric_mask: torch.Tensor,
        rubric_span: Optional[torch.Tensor], 
        label_id: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        """
        Forward method for the Pointer Rubric BiEncoder Model.
        
        Args:
            ans_input_ids: Input IDs for student answer
            ans_attention_mask: Attention mask for student answer
            rubric_input_ids: Input IDs for rubrics with shape (batch_size, max_rubrics, max_length)
            rubric_attention_mask: Attention mask for rubrics with shape (batch_size, max_rubrics, max_length)
            rubric_mask: Binary mask indicating valid rubrics (batch_size, max_rubrics)
            label_id: Labels for the correct rubric index
        """
        # Encode student answer
        ans_outputs = self.ans_encoder(
            input_ids=ans_input_ids,
            attention_mask=ans_attention_mask
        )
        ans_embeddings = ans_outputs.last_hidden_state[:, 0, :]  # Use [CLS] token
        
        # Process rubrics
        rubric_outputs = self.rubric_encoder(input_ids=rubric_input_ids, attention_mask=rubric_attention_mask)
        B, R = rubric_span.size(0), rubric_span.size(1)
        H = rubric_outputs.last_hidden_state.size(-1)
        rubric_embeddings = torch.zeros((B, R, H), dtype=rubric_outputs.last_hidden_state.dtype, device=rubric_outputs.last_hidden_state.device)
        for i in range(R):
            starts, ends = rubric_span[:, i, 0], rubric_span[:, i, 1]
            emb = torch.stack([
                rubric_outputs.last_hidden_state[b, starts[b]:ends[b]].mean(dim=0) 
                if rubric_mask[b, i] else torch.zeros(H, dtype=rubric_outputs.last_hidden_state.dtype, device=rubric_outputs.last_hidden_state.device)
                for b in range(B)
            ])
            rubric_embeddings[:, i, :] = emb
        # Compute pointer scores
        ans_embeddings = ans_embeddings.unsqueeze(1).expand(-1, R, -1)
        scores = self.pointer_head(rubric_embeddings, ans_embeddings).squeeze(-1)
        scores = scores.masked_fill(~rubric_mask, float('-inf'))  # Apply mask to scores
        logits = F.softmax(scores, dim=-1)  # Convert scores to probabilities
        loss = None
        if label_id is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), label_id.view(-1))
        return ModelOutput(logits=logits, loss=loss)
class AsagSNet(AsagXNet):
    """
    Sentence Transformer model for ASAG that computes cosine similarity between
    separately encoded student answers and rubrics.
    """
    def __init__(
        self, 
        base_model: str, 
        n_labels: int = 1, 
        freeze_layers: int = 0, 
        freeze_embeddings: bool = False, 
        label_weights: Optional[torch.Tensor] = None,
        use_lora: bool = False
    ):
        super().__init__(
            base_model=base_model,
            n_labels=n_labels,
            freeze_layers=freeze_layers,
            freeze_embeddings=freeze_embeddings,
            label_weights=label_weights,
            use_lora=use_lora
        )

    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Mean pooling or use [CLS] token for embeddings
        embeddings = mean_pooling(outputs.last_hidden_state, attention_mask)
        # embeddings =  outputs.last_hidden_state[:, 0, :]

        return embeddings
        
    def forward(
        self, 
        input_ids_a: torch.Tensor,  # Student answer
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,  # Rubric
        attention_mask_b: torch.Tensor,
        label_id: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        
        # Encode student answer
        embeddings_a = self.encode(input_ids_a, attention_mask_a)
        # Encode rubric
        embeddings_b = self.encode(input_ids_b, attention_mask_b)
        
        # Normalize embeddings for cosine similarity
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
        loss = None
        # h = torch.concat([embeddings_a, embeddings_b], dim=-1)
        # logits = self.classifier(h)
        logits = cosine_similarity(embeddings_a, embeddings_b, dim=-1).unsqueeze(1)
        logits = sigmoid(logits)  
        if label_id is not None:
            loss = self.get_loss_mse(logits, label_id)
        return ModelOutput(logits=logits, loss=loss)

        
if __name__ == "__main__":
    # Test small T5 model initialization
    from data_prep_alice import AliceRubricDataset 
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    import torch 
    model = AsagXNet(
        base_model="bert-base-uncased",
        n_labels=2,
        freeze_layers=0,
        use_lora=True
    )
    input_ids = torch.randint(0, 1000, (2, 10))  # Batch of 2 sequences of length 10
    attention_mask = torch.ones((2, 10), dtype=torch.long)
    label_id = torch.tensor([1, 0])  # Example labels for binary classification
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, label_id=label_id)
    print(outputs.logits)  # Should print logits for each sequence
    print(outputs.loss)  # Should print the loss value