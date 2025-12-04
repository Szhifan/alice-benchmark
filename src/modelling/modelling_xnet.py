from typing import Optional
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import logging
from modelling.modelling_utils import Pooler, BaseAsagModel

logger = logging.get_logger(__name__)


class AsagXnet(BaseAsagModel):
    def __init__(self, config, lora_config=None, bnb_config=None):
        super().__init__(config, lora_config=lora_config, bnb_config=bnb_config)
        
        self.num_labels = config.num_labels
        self.pooler = Pooler(pool_type=config.pool_type)
        self.use_token_type_ids = getattr(config, 'use_token_type_ids', True)
        self.score = nn.Linear(config.hidden_size, 1, bias=False)

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
        
        return mask

    def forward(
        self,
        input_ids: torch.LongTensor,        # [B, R, S]
        attention_mask: torch.Tensor,       # [B, R, S]
        token_type_ids: Optional[torch.Tensor] = None,  # [B, R, S]
        num_rubrics: Optional[torch.Tensor] = None,     # [B]
        labels: Optional[torch.LongTensor] = None,      # [B], index of correct rubric
        tau: float = 1.0,
    ) -> SequenceClassifierOutput:
        B, R, S = input_ids.shape

        # Build rubric mask
        rubric_mask = self._build_rubric_mask(
            num_rubrics, B, R, input_ids.device, 
            labels=labels if self.training else None,
            dtype=torch.long
        )

        # Flatten batch+rubric for the encoder: [B*R, S]
        flat_input_ids = input_ids.reshape(B * R, S)
        flat_attention_mask = attention_mask.reshape(B * R, S)
        flat_token_type_ids = None
        
        if self.use_token_type_ids and (token_type_ids is not None):
            flat_token_type_ids = token_type_ids.reshape(B * R, S)

        # ---- Encode ----
        transformer_outputs = self.get_encoder_outputs(
            flat_input_ids, 
            flat_attention_mask, 
            flat_token_type_ids
        )

        # ---- Pool ----
        if hasattr(transformer_outputs, "pooler_output") and transformer_outputs.pooler_output is not None:
            pooled_output = transformer_outputs.pooler_output  # [B*R, H]
        else:
            pooled_output = self.pooler(transformer_outputs.last_hidden_state, flat_attention_mask)  # [B*R, H]

        # ---- Score ----
        logits = self.score(pooled_output).squeeze(-1).reshape(B, R) 

        loss = None
        if labels is not None:
            loss = self.listwise_loss(logits, rubric_mask, labels, tau=tau)

        return SequenceClassifierOutput(loss=loss, logits=logits)