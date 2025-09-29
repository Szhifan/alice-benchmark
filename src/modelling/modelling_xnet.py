from typing import List, Optional, Tuple, Union
from transformers import AutoModel
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from .modelling_utils import Pooler, Network_Backbone
from transformers.utils import logging

logger = logging.get_logger(__name__)


class AsagXnet(Network_Backbone):
    def __init__(self, config, lora_config=None, bnb_config=None):
        super().__init__(config=config, lora_config=lora_config, bnb_config=bnb_config)
        self.config = config
        self.num_labels = config.num_labels
        self.pooler = Pooler(pool_type=config.pool_type)
        self.use_token_type_ids = getattr(config, 'use_token_type_ids', True)
        self.score = nn.Linear(config.hidden_size, 1, bias=False)
        
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
        input_ids: Optional[torch.LongTensor] = None,           # [B, R, S] for batch format or [B, S] for pair format
        attention_mask: Optional[torch.Tensor] = None,          # [B, R, S] or [B, S]
        token_type_ids: Optional[torch.Tensor] = None,          # [B, R, S] or [B, S]
        num_rubrics: Optional[torch.Tensor] = None,             # [B] - only used in batch format
        labels: Optional[torch.LongTensor] = None,              # [B] for batch format (rubric index) or [B] for pair format (binary)
        tau: float = 1.0,                                       # only used in batch format
        mask_prob: float = 0.0,                                 # only used in batch format,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        
        # Check if we're using the new batch format [B, R, S] or old pair format [B, S]
        if input_ids.dim() == 3:  # Batch format: [B, R, S]
            return self._forward_batch(
                input_ids, attention_mask, token_type_ids, 
                num_rubrics, labels, tau, mask_prob
            )
        else:  # Pair format: [B, S] - backward compatibility
            return self._forward_pair(
                input_ids, attention_mask, token_type_ids, labels
            )

    def _forward_batch(
        self,
        input_ids: torch.LongTensor,        # [B, R, S]
        attention_mask: torch.Tensor,       # [B, R, S]
        token_type_ids: Optional[torch.Tensor] = None,  # [B, R, S]
        num_rubrics: Optional[torch.Tensor] = None,     # [B]
        labels: Optional[torch.LongTensor] = None,      # [B], index of correct rubric
        tau: float = 1.0,
        mask_prob: float = 0.0,
    ) -> SequenceClassifierOutput:
        B, R, S = input_ids.shape

        # Build rubric mask (with probabilistic random negative masking)
        rubric_mask = self._build_rubric_mask(
            num_rubrics, B, R, input_ids.device, 
            labels=labels if self.training else None,  # 只在训练时传入labels
            mask_prob=mask_prob if self.training else 0.0,  # 只在训练时启用
            dtype=torch.long
        )

        # Flatten batch+rubric for the encoder: [B*R, S]
        flat_inputs = {
            "input_ids": input_ids.reshape(B * R, S),
            "attention_mask": attention_mask.reshape(B * R, S),
        }
        if self.use_token_type_ids and (token_type_ids is not None):
            flat_inputs["token_type_ids"] = token_type_ids.reshape(B * R, S)

        # ---- Encode ----
        # NOTE: when GC is enabled, parent already set use_cache=False and non-reentrant GC.
        transformer_outputs = self.encoder(**flat_inputs)

        # ---- Pool ----
        if hasattr(transformer_outputs, "pooler_output") and transformer_outputs.pooler_output is not None:
            pooled_output = transformer_outputs.pooler_output  # [B*R, H]
        else:
            pooled_output = self.pooler(transformer_outputs.last_hidden_state, flat_inputs["attention_mask"])  # [B*R, H]

        # ---- Score ----
        logits = self.score(pooled_output).squeeze(-1).reshape(B, R) 

        loss = None
        if labels is not None:
            loss = self.listwise_loss(logits, rubric_mask, labels, tau=tau)

        return SequenceClassifierOutput(loss=loss, logits=logits)

    def _forward_pair(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> SequenceClassifierOutput:
        """Pair-wise forward method for backward compatibility"""
        # For paired input, the labels are expected to be binary (0/1)
        assert labels is None or ((labels == 0) | (labels == 1)).all(), "For pair format, labels must be binary (0/1)."
        
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        if self.use_token_type_ids and token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids

        transformer_outputs = self.encoder(**inputs)
        
        # ---- Pool ----
        if hasattr(transformer_outputs, "pooler_output") and transformer_outputs.pooler_output is not None:
            pooled_output = transformer_outputs.pooler_output
        else:
            pooled_output = self.pooler(transformer_outputs.last_hidden_state, attention_mask)

        # ---- Score ----
        logits = self.score(pooled_output)
        
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
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)

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
        assert (pos_idx == -1).any() or (pos_mask == 1).all(), "pos_idx must refer to valid rubrics or be -1."

        return nn.CrossEntropyLoss()(masked_logits, pos_idx)