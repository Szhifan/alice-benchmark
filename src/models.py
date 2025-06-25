import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
import torch
from transformers import AutoModel
from torch.nn import CrossEntropyLoss, Bilinear
from torch.nn.functional import sigmoid, cosine_similarity
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
    ):
        super().__init__()
        self.model_name = model_name 
        self.encoder = AutoModel.from_pretrained(model_name) 
        hidden_size = self.encoder.config.hidden_size
        self.label_weights = label_weights
        self.num_labels = num_labels 
        self.classifier = ClassificationHead(hidden_size, self.num_labels)
        
        freeze_bert_layers(self.encoder, freeze_layers)
        if freeze_embeddings:
            freeze_bert_embeddings(self.encoder)
    def get_loss_ce(self, logits: torch.Tensor, label_id: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross-entropy loss.
        """
        self.label_weights = self.label_weights.to(logits.device) if self.label_weights is not None else None
        loss_fct = CrossEntropyLoss(weight=self.label_weights)
        return loss_fct(logits.view(-1, self.num_labels), label_id.view(-1)), logits
    def get_loss_mse(self, logits: torch.Tensor, label_id: torch.Tensor) -> torch.Tensor:
        """
        Compute the mean squared error loss.
        """
        loss_fct = nn.MSELoss()
        prob =  sigmoid(logits)
        label_id = label_id.float()
        return loss_fct(prob.view(-1, self.num_labels), label_id.view(-1, self.num_labels)), prob
    def get_loss(self, logits: torch.Tensor, label_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the loss based on the number of labels.
        """
        if self.num_labels > 1:
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

class PointerRubricModel(nn.Module):
    """
    Pointer Rubric Model for ASAG.
    This model uses a T5 encoder to process the input and a pointer mechanism to select the rubric.
    """
    def __init__(self, model_name: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
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
    Pointer Rubric BiEncoder Model for ASAG.
    This model uses separate encoders to process the student answer and rubrics,
    and a pointer mechanism to select the relevant rubric.
    """
    def __init__(self, model_name: str, share_encoders: bool = True):
        super().__init__()
        if share_encoders:
            self.ans_encoder = self.rubric_encoder = AutoModel.from_pretrained(model_name)
        else:
            self.ans_encoder = AutoModel.from_pretrained(model_name)
            self.rubric_encoder = AutoModel.from_pretrained(model_name)
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
class AsagSentenceTransformer(AsagCrossEncoder):
    """
    Sentence Transformer model for ASAG that computes cosine similarity between
    separately encoded student answers and rubrics.
    """
    def __init__(
        self, 
        model_name: str, 
        num_labels: int = 2, 
        freeze_layers: int = 0, 
        freeze_embeddings: bool = False, 
        label_weights: Optional[torch.Tensor] = None, 
    ):
        super().__init__(
            model_name=model_name,
            num_labels=num_labels,
            freeze_layers=freeze_layers,
            freeze_embeddings=freeze_embeddings,
            label_weights=label_weights
        )
        d_in_features = self.encoder.config.hidden_size * 3  # Concatenated features
        self.classifier = nn.Linear(d_in_features, num_labels)  # For classification tasks
    
    def encode(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        # Mean pooling to get sentence representation
        embeddings = mean_pooling(outputs, attention_mask)
    
        return embeddings
    def emb_features(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> torch.Tensor:
        features = torch.cat([emb_a, emb_b,emb_a - emb_b], dim=-1)
        return features
    def get_loss_ce(self, logits, label_id):
        loss_fct = CrossEntropyLoss(weight=self.label_weights) if self.label_weights is not None else CrossEntropyLoss()
        return loss_fct(logits.view(-1, self.num_labels), label_id.view(-1)), logits
        
    def forward(
        self, 
        input_ids_a: torch.Tensor,  # Student answer
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,  # Rubric
        attention_mask_b: torch.Tensor,
        token_type_ids_a: Optional[torch.Tensor] = None,
        token_type_ids_b: Optional[torch.Tensor] = None,
        label_id: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        
        # Encode student answer
        embeddings_a = self.encode(input_ids_a, attention_mask_a, token_type_ids_a)
        # Encode rubric
        embeddings_b = self.encode(input_ids_b, attention_mask_b, token_type_ids_b)
        
        # Normalize embeddings for cosine similarity
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
        if self.num_labels == 1:
            # For regression task, compute cosine similarity
            cosine_sim = cosine_similarity(embeddings_a, embeddings_b, dim=1)
            logits = cosine_sim.unsqueeze(1)  # Shape: (B, 1)
        else:
            # classification task
            features = self.emb_features(embeddings_a, embeddings_b)
            logits = self.classifier(features)  # Shape: (B, num_labels)
        loss = None
        if label_id is not None:
            # Cross-entropy loss for classification
            loss, logits = self.get_loss_ce(logits, label_id)
        return ModelOutput(logits=logits, loss=loss)
        
if __name__ == "__main__":
    import torch 
    from data_prep_asap import AsapRubric
    from data_prep_alice import AliceRubricDataset
    from torch.utils.data import DataLoader
    dts = AliceRubricDataset()
    tokenizer = get_tokenizer("bert-base-multilingual-uncased")
    
    dts.get_encoding(tokenizer)
    test_ds = dts.test_ua
    loader = DataLoader(
        test_ds, 
        batch_size=4, 
        collate_fn=dts.collate_fn,
        shuffle=True,
    )
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AsagCrossEncoder(
        model_name="bert-base-multilingual-uncased", use_ce_loss=False
    ).to(device)
    for batch, meta in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.to(device)
        label_id = batch["label_id"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            label_id=label_id
        )
        logits = outputs.logits
        loss = outputs.loss
        print(f"Logits: {logits}, Loss: {loss}")
        break