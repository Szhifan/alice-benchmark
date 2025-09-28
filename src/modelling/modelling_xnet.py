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


class AsagXnet(Network_Backbone):
    def __init__(self, config, lora_config=None, bnb_config=None, enable_gc: bool = False):
        super().__init__(config=config, lora_config=lora_config, bnb_config=bnb_config)
        self.config = config
        self.num_labels = config.num_labels
        self.pooler = Pooler(pool_type=config.pool_type)
        self.use_token_type_ids = getattr(config, 'use_token_type_ids', True)

        # Classification head - outputs scalar score per rubric
        self.score = nn.Linear(config.hidden_size, 1, bias=False)

        # ---- GC integration: keep cache off only when training with GC ----
        if enable_gc:
            # parent implements: use_cache=False, non-reentrant, input grads for LoRA
            super().enable_gradient_checkpointing()

    # Keep cache/use_cache consistent with train/eval modes:
    def train(self, mode: bool = True):
        super().train(mode)
        # If we are training and GC is enabled -> keep cache off; otherwise allow cache
        if mode:
            if hasattr(self.encoder, "is_gradient_checkpointing") and self.encoder.is_gradient_checkpointing:
                if hasattr(self.encoder, "config"):
                    self.encoder.config.use_cache = False
        else:
            # eval mode: disable GC and re-enable cache for speed
            try:
                self.encoder.gradient_checkpointing_disable()
            except AttributeError:
                pass
            if hasattr(self.encoder, "config"):
                self.encoder.config.use_cache = True
        return self

    # Convenience passthroughs if you want to toggle GC from outside
    def enable_gradient_checkpointing(self):
        super().enable_gradient_checkpointing()

    def disable_gradient_checkpointing(self):
        super().disable_gradient_checkpointing()

    @torch.no_grad()
    def _build_rubric_mask(self, num_rubrics: Optional[torch.Tensor], B: int, R: int, device, dtype=torch.long):
        """
        Vectorized mask builder: 1 for valid rubrics, 0 for padding.
        num_rubrics: [B] or None
        """
        if num_rubrics is None:
            return torch.ones(B, R, device=device, dtype=dtype)
        # num_rubrics may come as int64 on CPU; ensure device/dtype
        nr = num_rubrics.to(device=device)
        # [B, R] where j < nr[i]
        ar = torch.arange(R, device=device).unsqueeze(0).expand(B, R)
        return (ar < nr.unsqueeze(1)).to(dtype)

    def forward(
        self,
        input_ids: torch.LongTensor = None,        # [B, R, S]
        attention_mask: torch.Tensor = None,       # [B, R, S]
        token_type_ids: Optional[torch.Tensor] = None,  # [B, R, S]
        num_rubrics: Optional[torch.Tensor] = None,     # [B]
        labels: Optional[torch.LongTensor] = None,      # [B], index of correct rubric
        tau: float = 1.0,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        B, R, S = input_ids.shape

        # Build rubric mask (vectorized)
        rubric_mask = self._build_rubric_mask(num_rubrics, B, R, input_ids.device, dtype=torch.long)

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
            # fall back to your custom pooler ([CLS]/mean etc.)
            pooled_output = self.pooler(transformer_outputs.last_hidden_state, flat_inputs["attention_mask"])  # [B*R, H]

        # ---- Score ----
        logits = self.score(pooled_output).squeeze(-1).reshape(B, R)  # [B, R]

        loss = None
        if labels is not None:
            loss = self.listwise_loss(logits, rubric_mask, labels, tau=tau)

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
        assert (pos_mask == 1).all(), "pos_idx must refer to valid rubrics."

        return nn.CrossEntropyLoss()(masked_logits, pos_idx)
