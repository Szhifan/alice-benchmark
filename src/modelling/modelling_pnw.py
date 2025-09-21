from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from .modelling_utils import Pooler, Network_Backbone
from transformers.utils import logging

logger = logging.get_logger(__name__)

class RubricPointer(Network_Backbone):
    def __init__(self, config, lora_config=None, emb_type=None, bnb_config=None):
        super().__init__(config=config, lora_config=lora_config, bnb_config=bnb_config)
        self.config = config
        self.num_labels = config.num_labels
        self.pooler = Pooler(pool_type=getattr(config, 'pool_type', 'cls'))
        
        # Transformer layer for cross-attention between rubrics and answer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=getattr(config, 'num_attention_heads', 8),
            dim_feedforward=getattr(config, 'intermediate_size', config.hidden_size * 4),
            dropout=getattr(config, 'hidden_dropout_prob', 0.1),
            batch_first=True
        )
        
        # Classification head for each rubric
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        
        # Optional: final aggregation layer
        self.use_aggregation = getattr(config, 'use_aggregation', True)
        if self.use_aggregation:
            self.aggregation_layer = nn.Linear(config.hidden_size, config.hidden_size)
            self.aggregation_pooler = Pooler(pool_type='avg')
# ...existing code...

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
        seq_mask: torch.Tensor = None,
        n_seq: torch.Tensor = None,
        n_rubrics: torch.Tensor = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        """
        Forward pass for pointer network.
        
        Args:
            input_ids: (batch_size, max_n_seq, max_seq_len)
            attention_mask: (batch_size, max_n_seq, max_seq_len)
            token_type_ids: (batch_size, max_n_seq, max_seq_len) - optional
            seq_mask: (batch_size, max_n_seq) - mask for valid sequences
            n_seq: (batch_size,) - number of sequences per example
            n_rubrics: (batch_size,) - number of rubrics per example
            labels: (batch_size,) - target rubric level (0-indexed)
        """
        batch_size, max_n_seq, max_seq_len = input_ids.shape
        
        # Reshape for encoder processing
        input_ids_flat = input_ids.view(-1, max_seq_len)
        attention_mask_flat = attention_mask.view(-1, max_seq_len)
        
        # Handle token_type_ids if present
        token_type_ids_flat = None
        if token_type_ids is not None:
            token_type_ids_flat = token_type_ids.view(-1, max_seq_len)
        
        # Encode all sequences
        encoder_outputs = self.encoder(
            input_ids=input_ids_flat,
            attention_mask=attention_mask_flat,
            token_type_ids=token_type_ids_flat
        )
        
        # Pool each sequence
        pooled_outputs = self.pooler(
            encoder_outputs.last_hidden_state, 
            attention_mask_flat
        )  # (batch_size * max_n_seq, hidden_size)
        
        # Reshape back to batch format
        pooled_outputs = pooled_outputs.view(batch_size, max_n_seq, -1)
        
        # Apply transformer layer for cross-attention
        # Create attention mask for transformer (True for valid positions)
        transformer_mask = ~seq_mask  # Invert for transformer (False = attend)
        
        # Apply transformer layer
        transformed_outputs = self.transformer_layer(
            pooled_outputs,
            src_key_padding_mask=transformer_mask
        )  # (batch_size, max_n_seq, hidden_size)
        
        max_rubrics = n_rubrics.max().item()
        
        # Create padded logits tensor for all examples
        all_rubric_logits = torch.full(
            (batch_size, max_rubrics), 
            float('-inf'), 
            device=transformed_outputs.device,
            dtype=transformed_outputs.dtype
        )
        for i in range(batch_size):
            n_rubric = n_rubrics[i].item()
            # Extract rubric representations (assuming rubrics come first)
            rubric_representations = transformed_outputs[i, :n_rubric]  # (n_rubric, hidden_size)
            
            # Map each rubric to a score
            rubric_scores = self.classifier(rubric_representations).squeeze(-1)  # (n_rubric,)
            
            # Store in padded tensor
            all_rubric_logits[i, :n_rubric] = rubric_scores
        
        # Create attention mask for rubrics
        rubric_mask = torch.arange(max_rubrics, device=n_rubrics.device)[None, :] < n_rubrics[:, None]
        

        masked_logits = all_rubric_logits.masked_fill(~rubric_mask, float('-inf'))
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            labels = labels.to(masked_logits.device)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(masked_logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=masked_logits,  # (batch_size, max_rubrics)
            hidden_states=transformed_outputs if not self.training else None
        )

    def predict_rubric_scores(self, **inputs):
        """
        Convenience method to get rubric scores as probabilities.
        Returns softmax probabilities over valid rubrics for each example.
        """
        with torch.no_grad():
            outputs = self.forward(**inputs)
            logits = outputs.logits  # (batch_size, max_rubrics)
            n_rubrics = inputs['n_rubrics']
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)
            
            # Extract valid probabilities for each example
            batch_probs = []
            for i in range(logits.size(0)):
                n_rubric = n_rubrics[i].item()
                valid_probs = probs[i, :n_rubric]
                batch_probs.append(valid_probs)
            
            return batch_probs

