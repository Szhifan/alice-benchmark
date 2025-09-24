from typing import List, Optional, Tuple, Union
from transformers import AutoModel
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from .modelling_utils import Pooler, Network_Backbone
from transformers.utils import logging

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "LlamaConfig"


class XnetSoftmaxCE(Network_Backbone):
    def __init__(self, config, lora_config=None, bnb_config=None):
        super().__init__(config=config, lora_config=lora_config, bnb_config=bnb_config)
        self.config = config
        self.num_labels = config.num_labels
        self.pooler = Pooler(pool_type=config.pool_type)
        self.use_token_type_ids = getattr(config, 'use_token_type_ids', True)
        
        # Classification head - outputs scalar score per rubric
        self.head = nn.Linear(config.hidden_size, 1, bias=False)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        token_type_ids: Optional[torch.Tensor] = None,
        num_rubrics: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        tau: float = 1.0,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        
        # input_ids: [B, R, S], attention_mask: [B, R, S]
        B, R, S = input_ids.shape
        
        # Create rubric mask from num_rubrics
        if num_rubrics is not None:
            # Create mask: 1 for valid rubrics, 0 for padding
            rubric_mask = torch.zeros(B, R, device=input_ids.device, dtype=torch.long)
            for i, num_rub in enumerate(num_rubrics):
                rubric_mask[i, :num_rub] = 1
        else:
            # If num_rubrics not provided, assume all are valid
            rubric_mask = torch.ones(B, R, device=input_ids.device, dtype=torch.long)
        
        # Flatten for encoder: [B*R, S]
        flat_inputs = {
            "input_ids": input_ids.view(B*R, S),
            "attention_mask": attention_mask.view(B*R, S),
        }
        if self.use_token_type_ids and token_type_ids is not None:
            flat_inputs["token_type_ids"] = token_type_ids.view(B*R, S)
        
        # Encode through transformer
        transformer_outputs = self.encoder(**flat_inputs)
        
        # Pool representations: [B*R, H]
        if hasattr(transformer_outputs, "pooler_output") and transformer_outputs.pooler_output is not None:
            pooled_output = transformer_outputs.pooler_output
        else:
            # Use pooler if available, otherwise use [CLS] token or mean pooling
            pooled_output = self.pooler(transformer_outputs.last_hidden_state, flat_inputs["attention_mask"])
        
        # Get logits: [B*R, 1] -> [B*R] -> [B, R]
        logits = self.head(pooled_output).squeeze(-1)  # [B*R]
        logits = logits.view(B, R)  # [B, R]
        
        loss = None
        if labels is not None:
            loss = self.listwise_loss(logits, rubric_mask, labels, tau=tau)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits
        )
    
    def listwise_loss(
        self, 
        logits: torch.Tensor, 
        rubric_mask: torch.Tensor,
        pos_idx: torch.Tensor, 
        tau: float = 1.0
    ) -> torch.Tensor:
        """
        Listwise loss for ranking rubrics.
        
        Args:
            logits: [B, R] - scores for each rubric
            rubric_mask: [B, R] - 1 for valid rubrics, 0 for padding
            pos_idx: [B] - indices of correct rubrics (original labels)
            tau: temperature parameter for softmax
            
        Returns:
            loss: scalar loss value
        """
        # Ensure every sample has at least one valid rubric
        assert (rubric_mask.sum(dim=1) > 0).all(), "Every sample needs at least one valid rubric."
        
        # Apply temperature scaling
        scaled_logits = logits / tau
        
        # Mask out invalid rubrics with very negative values
        very_neg = torch.finfo(logits.dtype).min
        masked_logits = scaled_logits.masked_fill(rubric_mask == 0, very_neg)
        
        # Ensure pos_idx refers to valid rubrics
        pos_mask = rubric_mask.gather(1, pos_idx.view(-1, 1)).squeeze(1)
        assert (pos_mask == 1).all(), "pos_idx must refer to valid rubrics."
        
        # Cross-entropy loss
        ce_loss = nn.CrossEntropyLoss()
        return ce_loss(masked_logits, pos_idx)