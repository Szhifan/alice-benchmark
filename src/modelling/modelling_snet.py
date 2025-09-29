# Built upon the huggingface implementation 

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


class AsagSNet(Network_Backbone):
    def __init__(self, config, lora_config=None, emb_type=None, bnb_config=None):
        super().__init__(config=config, lora_config=lora_config, bnb_config=bnb_config)
        self.config = config
        self.num_labels = config.num_labels
        self.emb_type = emb_type if emb_type is not None else 'diffABS'
        self.pooler = Pooler(pool_type=config.pool_type)
        config.emb_type = self.emb_type
        if self.emb_type in['diff','diffABS']:
            input_size = config.hidden_size
        elif self.emb_type in ['n-o','n-diffABS']:
            input_size = config.hidden_size*2
        elif self.emb_type in ['n-diffABS-o']:
            input_size = config.hidden_size*3
        else:
            raise ValueError("invalid emb_type")
        self.score = nn.Linear(input_size, 1, bias=False)  # Change to output 1 score per comparison
        self.use_token_type_ids = getattr(config, 'use_token_type_ids', True)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

    def gradient_checkpointing_disable(self):
        self.encoder.gradient_checkpointing_disable()

    @torch.no_grad()
    def _build_rubric_mask(self, num_rubrics: Optional[torch.Tensor], B: int, R: int, device, 
                      labels: Optional[torch.Tensor] = None, 
                      mask_prob: float = 0.0, dtype=torch.long):
        """
        Vectorized mask builder: 1 for valid rubrics, 0 for padding.
        num_rubrics: [B] or None
        labels: [B] correct rubric indices (needed for random masking)
        mask_prob: probability of applying random negative masking per sample
        """
        if num_rubrics is None:
            mask = torch.ones(B, R, device=device, dtype=dtype)
        else:
            # num_rubrics may come as int64 on CPU; ensure device/dtype
            nr = num_rubrics.to(device=device)
            # [B, R] where j < nr[i]
            ar = torch.arange(R, device=device).unsqueeze(0).expand(B, R)
            mask = (ar < nr.unsqueeze(1)).to(dtype)
        
        # Apply random negative masking with specified probability
        if mask_prob > 0.0 and labels is not None:
            # Generate random decisions for each sample
            random_decisions = torch.rand(B, device=device) < mask_prob
            
            for i in range(B):
                if random_decisions[i]:
                    # Find valid negative rubrics (valid but not the correct one)
                    valid_mask = mask[i] == 1
                    negative_indices = torch.where(valid_mask & (torch.arange(R, device=device) != labels[i]))[0]
                    
                    if len(negative_indices) > 0:
                        random_idx = negative_indices[torch.randint(len(negative_indices), (1,))]
                        mask[i, random_idx] = 0
        
        return mask

    def forward(
        self,
        input_ids_a: Optional[torch.LongTensor] = None,        # [B, R, S] for batch format or [B, S] for pair format
        input_ids_b: Optional[torch.LongTensor] = None,        # [B, R, S] for batch format or [B, S] for pair format  
        attention_mask_a: Optional[torch.Tensor] = None,       # [B, R, S] or [B, S]
        attention_mask_b: Optional[torch.Tensor] = None,       # [B, R, S] or [B, S]
        token_type_ids_a: Optional[torch.Tensor] = None,       # [B, R, S] or [B, S]
        token_type_ids_b: Optional[torch.Tensor] = None,       # [B, R, S] or [B, S]
        num_rubrics: Optional[torch.Tensor] = None,            # [B] - only used in batch format
        labels: Optional[torch.LongTensor] = None,             # [B] for batch format or [B] for pair format
        tau: float = 1.0,                                      # only used in batch format
        mask_prob: float = 0.0,                                # only used in batch format
    ) -> Union[Tuple, SequenceClassifierOutput]:
        
        # Check if we're using the new batch format [B, R, S] or old pair format [B, S]
        if input_ids_a.dim() == 3:  # Batch format: [B, R, S]
            return self._forward_batch(
                input_ids_a, input_ids_b, attention_mask_a, attention_mask_b,
                token_type_ids_a, token_type_ids_b, num_rubrics, labels, tau
            )
        else:  # Pair format: [B, S] - backward compatibility
            return self._forward_pair(
                input_ids_a, input_ids_b, attention_mask_a, attention_mask_b,
                token_type_ids_a, token_type_ids_b, labels
            )

    def _forward_batch(
        self,
        input_ids_a: torch.LongTensor,        # [B, R, S]
        input_ids_b: torch.LongTensor,        # [B, R, S]
        attention_mask_a: torch.Tensor,       # [B, R, S]
        attention_mask_b: torch.Tensor,       # [B, R, S]
        token_type_ids_a: Optional[torch.Tensor] = None,  # [B, R, S]
        token_type_ids_b: Optional[torch.Tensor] = None,  # [B, R, S]
        num_rubrics: Optional[torch.Tensor] = None,       # [B]
        labels: Optional[torch.LongTensor] = None,        # [B] - index of correct rubric
        tau: float = 1.0,
        mask_prob: float = 0.0,
    ) -> SequenceClassifierOutput:
        B, R, S = input_ids_a.shape

        # Build rubric mask (vectorized)
        rubric_mask = self._build_rubric_mask(num_rubrics, B, R, input_ids_a.device, dtype=torch.long)

        # Flatten batch+rubric for the encoder: [B*R, S]
        flat_inputs_a = {
            "input_ids": input_ids_a.reshape(B * R, S),
            "attention_mask": attention_mask_a.reshape(B * R, S),
        }
        flat_inputs_b = {
            "input_ids": input_ids_b.reshape(B * R, S),
            "attention_mask": attention_mask_b.reshape(B * R, S),
        }
        
        if self.use_token_type_ids:
            if token_type_ids_a is not None:
                flat_inputs_a["token_type_ids"] = token_type_ids_a.reshape(B * R, S)
            if token_type_ids_b is not None:
                flat_inputs_b["token_type_ids"] = token_type_ids_b.reshape(B * R, S)

        # ---- Encode both sequences ----
        transformer_outputs_a = self.encoder(**flat_inputs_a)
        transformer_outputs_b = self.encoder(**flat_inputs_b)

        # ---- Pool ----
        if hasattr(transformer_outputs_a, "pooler_output") and transformer_outputs_a.pooler_output is not None:
            pooled_a = transformer_outputs_a.pooler_output  # [B*R, H]
        else:
            pooled_a = self.pooler(transformer_outputs_a.last_hidden_state, flat_inputs_a["attention_mask"])  # [B*R, H]
            
        if hasattr(transformer_outputs_b, "pooler_output") and transformer_outputs_b.pooler_output is not None:
            pooled_b = transformer_outputs_b.pooler_output  # [B*R, H]
        else:
            pooled_b = self.pooler(transformer_outputs_b.last_hidden_state, flat_inputs_b["attention_mask"])  # [B*R, H]

        # ---- Combine embeddings based on emb_type ----
        if self.emb_type == 'diff':
            hidden_states = torch.as_tensor(pooled_a - pooled_b)
        elif self.emb_type == 'diffABS':
            hidden_states = torch.abs(torch.as_tensor(pooled_a - pooled_b))
        elif self.emb_type == 'n-diffABS':
            diff = torch.abs(torch.as_tensor(pooled_a - pooled_b))
            hidden_states = torch.cat((pooled_a, diff), 1)
        elif self.emb_type == 'n-diffABS-o':
            diff = torch.abs(torch.as_tensor(pooled_a - pooled_b))
            hidden_states = torch.cat((pooled_a, diff, pooled_b), 1)
        elif self.emb_type == 'n-o':
            hidden_states = torch.cat((pooled_a, pooled_b), 1)

        # ---- Score ----
        logits = self.score(hidden_states).squeeze(-1).reshape(B, R)  # [B, R]

        loss = None
        if labels is not None:
            loss = self.listwise_loss(logits, rubric_mask, labels, tau=tau)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def _forward_pair(
        self,
        input_ids_a: torch.LongTensor,
        input_ids_b: torch.LongTensor,
        attention_mask_a: torch.Tensor,
        attention_mask_b: torch.Tensor,
        token_type_ids_a: Optional[torch.Tensor] = None,
        token_type_ids_b: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> SequenceClassifierOutput:
        """Original pair-wise forward method for backward compatibility"""
        # for paired input, the labels are expected to be binary (0/1)
        assert labels is None or ((labels == 0) | (labels == 1)).all(), "For pair format, labels must be binary (0/1)."
        inputs_a = {
            "input_ids": input_ids_a,
            "attention_mask": attention_mask_a,
        }
        inputs_b = {
            "input_ids": input_ids_b,
            "attention_mask": attention_mask_b,
        }
        
        if self.use_token_type_ids and token_type_ids_a is not None:
            inputs_a["token_type_ids"] = token_type_ids_a
        if self.use_token_type_ids and token_type_ids_b is not None:
            inputs_b["token_type_ids"] = token_type_ids_b

        transformer_outputs_a = self.encoder(**inputs_a)
        transformer_outputs_b = self.encoder(**inputs_b)
        
        pooled_a = self.pooler(transformer_outputs_a.last_hidden_state, attention_mask_a)
        pooled_b = self.pooler(transformer_outputs_b.last_hidden_state, attention_mask_b)

        if self.emb_type == 'diff':
            hidden_states = torch.as_tensor(pooled_a - pooled_b)
        elif self.emb_type == 'diffABS':
            hidden_states = torch.abs(torch.as_tensor(pooled_a - pooled_b))
        elif self.emb_type == 'n-diffABS':
            diff = torch.abs(torch.as_tensor(pooled_a - pooled_b))
            hidden_states = torch.cat((pooled_a, diff), 1)
        elif self.emb_type == 'n-diffABS-o':
            diff = torch.abs(torch.as_tensor(pooled_a - pooled_b))
            hidden_states = torch.cat((pooled_a, diff, pooled_b), 1)
        elif self.emb_type == 'n-o':
            hidden_states = torch.cat((pooled_a, pooled_b), 1)

        logits = self.score(hidden_states)
        pooled_logits = logits
        
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=pooled_logits)

    def listwise_loss(
        self,
        logits: torch.Tensor,          # [B, R]
        rubric_mask: torch.Tensor,     # [B, R] in {0,1}
        pos_idx: torch.Tensor,         # [B] index of correct rubric
        tau: float = 1.0
    ) -> torch.Tensor:
        """
        Listwise loss for ranking rubrics.
        """
        # Sanity: at least one valid rubric per example
        assert (rubric_mask.sum(dim=1) > 0).all(), "Every sample needs at least one valid rubric."

        # Temperature scaling
        scaled_logits = logits / tau

        # Mask invalid rubrics. Use a large negative (not -inf) to avoid NaNs in mixed precision.
        masked_logits = scaled_logits.masked_fill(rubric_mask == 0, -1e9)

        # Ensure pos_idx refers to valid rubrics
        pos_mask = rubric_mask.gather(1, pos_idx.view(-1, 1)).squeeze(1)
        assert (pos_mask == 1).all(), "pos_idx must refer to valid rubrics."

        return nn.CrossEntropyLoss()(masked_logits, pos_idx)