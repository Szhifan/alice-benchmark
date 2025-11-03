# Built upon the huggingface implementation 

from typing import List, Optional, Tuple, Union
from transformers import AutoModel, PreTrainedModel
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from .modelling_utils import Pooler
from transformers.utils import logging
import os

logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "LlamaConfig"


class AsagSNet(PreTrainedModel):
    def __init__(self, config, lora_config=None, emb_type=None, bnb_config=None):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_labels
        self.emb_type = emb_type if emb_type is not None else 'diffABS'
        self.pooler = Pooler(pool_type=config.pool_type)
        self.use_token_type_ids = getattr(config, 'use_token_type_ids', True)
        config.emb_type = self.emb_type

        # Scorer input dimension depends on emb_type
        if self.emb_type in ['diff', 'diffABS']:
            input_size = config.hidden_size
        elif self.emb_type in ['n-o', 'n-diffABS']:
            input_size = config.hidden_size * 2
        elif self.emb_type in ['n-diffABS-o']:
            input_size = config.hidden_size * 3
        else:
            raise ValueError("invalid emb_type")
        self.score = nn.Linear(input_size, 1, bias=False)

        # Backbone init (align with XNet)
        self.lora_config = lora_config
        self.bnb_config = bnb_config
        if bnb_config is not None:
            self.encoder = AutoModel.from_pretrained(
                config.base_model_name_or_path,
                quantization_config=bnb_config,
                config=config
            )
        else:
            self.encoder = AutoModel.from_pretrained(
                config.base_model_name_or_path,
                config=config
            )

    def init_peft(self):
        """Initialize PEFT model"""
        from peft import get_peft_model
        self.encoder = get_peft_model(self.encoder, self.lora_config)

    def _load_peft_adapter(self, ckpt_dir: str):
        """Load PEFT adapter weights"""
        from peft import PeftModel
        self.encoder = PeftModel.from_pretrained(self.encoder, ckpt_dir)

    @classmethod
    def from_pretrained(cls, model_path, config=None, lora_config=None, bnb_config=None, **kwargs):
        """
        Custom model loading logic that supports:
        1) Pure model (pytorch_model.bin)
        2) LoRA: adapter_model + non_peft_params.bin
        """
        model = cls(config, lora_config=lora_config, emb_type=getattr(config, "emb_type", None), bnb_config=bnb_config)

        adapter_file_pt = os.path.join(model_path, "adapter_model.bin")
        adapter_file_st = os.path.join(model_path, "adapter_model.safetensors")
        has_adapter = os.path.exists(adapter_file_pt) or os.path.exists(adapter_file_st)

        non_peft_file = os.path.join(model_path, "non_peft_params.bin")
        full_file = os.path.join(model_path, "pytorch_model.bin")

        if lora_config:
            if has_adapter:
                model._load_peft_adapter(model_path)
                logger.info(f"[LoRA Load] Successfully loaded LoRA adapter from {model_path}")
            else:
                logger.warning(f"[LoRA Load] Adapter model not found in {model_path}")

            if os.path.exists(non_peft_file):
                non_peft_state = torch.load(non_peft_file, map_location="cpu")
                missing, unexpected = model.load_state_dict(non_peft_state, strict=False)
                if missing:
                    logger.warning(f"[LoRA Load] Missing non_peft parameters: {missing}")
                if unexpected:
                    logger.warning(f"[LoRA Load] Unexpected non_peft parameters: {unexpected}")
            else:
                logger.warning(f"[LoRA Load] non_peft_params.bin not found in {model_path}")
        else:
            if os.path.exists(full_file):
                full_state = torch.load(full_file, map_location="cpu")
                missing, unexpected = model.load_state_dict(full_state, strict=False)
                if missing:
                    logger.warning(f"[Full Load] Missing parameters: {missing}")
                if unexpected:
                    logger.warning(f"[Full Load] Unexpected parameters: {unexpected}")
            else:
                logger.error(f"[Full Load] {full_file} not found")

        return model

    def save_pretrained(self, save_path, **kwargs):
        """
        Custom save logic that handles both LoRA and full model saving
        """
        os.makedirs(save_path, exist_ok=True)
        self.config.save_pretrained(save_path)

        if hasattr(self.encoder, 'save_pretrained') and hasattr(self.encoder, 'peft_config'):
            # Save PEFT adapter
            self.encoder.save_pretrained(save_path)
            # Save non-PEFT parameters (this module only)
            full_state = self.state_dict()
            to_remove = [k for k in full_state.keys() if k.startswith("encoder.")]
            for k in to_remove:
                full_state.pop(k, None)
            torch.save(full_state, os.path.join(save_path, "non_peft_params.bin"))
            logger.info(f"Saved LoRA adapter and non-PEFT parameters to {save_path}")
        else:
            torch.save(self.state_dict(), os.path.join(save_path, "pytorch_model.bin"))
            logger.info(f"Saved full model to {save_path}")

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
            nr = num_rubrics.to(device=device)
            ar = torch.arange(R, device=device).unsqueeze(0).expand(B, R)
            mask = (ar < nr.unsqueeze(1)).to(dtype)
        
        if mask_prob > 0.0 and labels is not None:
            random_decisions = torch.rand(B, device=device) < mask_prob
            for i in range(B):
                if random_decisions[i]:
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
        **kwargs
    ) -> Union[Tuple, SequenceClassifierOutput]:
        if input_ids_a.dim() == 3:  # Batch format: [B, R, S]
            return self._forward_batch(
                input_ids_a, input_ids_b, attention_mask_a, attention_mask_b,
                token_type_ids_a, token_type_ids_b, num_rubrics, labels, tau, mask_prob
            )
        else:  # Pair format: [B, S]
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

        # Build rubric mask; only apply random masking during training
        rubric_mask = self._build_rubric_mask(
            num_rubrics, B, R, input_ids_a.device,
            labels=labels if self.training else None,
            mask_prob=mask_prob if self.training else 0.0,
            dtype=torch.long
        )

        # Flatten for encoder: [B*R, S]
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

        # Encode
        transformer_outputs_a = self.encoder(**flat_inputs_a)
        transformer_outputs_b = self.encoder(**flat_inputs_b)

        # Pool with fallback to model's pooler_output if available
        if hasattr(transformer_outputs_a, "pooler_output") and transformer_outputs_a.pooler_output is not None:
            pooled_a = transformer_outputs_a.pooler_output
        else:
            pooled_a = self.pooler(transformer_outputs_a.last_hidden_state, flat_inputs_a["attention_mask"])

        if hasattr(transformer_outputs_b, "pooler_output") and transformer_outputs_b.pooler_output is not None:
            pooled_b = transformer_outputs_b.pooler_output
        else:
            pooled_b = self.pooler(transformer_outputs_b.last_hidden_state, flat_inputs_b["attention_mask"])

        # Combine embeddings per emb_type
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
        else:
            raise ValueError("invalid emb_type")

        # Score and reshape to [B, R]
        logits = self.score(hidden_states).squeeze(-1).reshape(B, R)

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
        """Pair-wise forward method for backward compatibility"""
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

        # Pool with fallback to model's pooler_output if available
        if hasattr(transformer_outputs_a, "pooler_output") and transformer_outputs_a.pooler_output is not None:
            pooled_a = transformer_outputs_a.pooler_output
        else:
            pooled_a = self.pooler(transformer_outputs_a.last_hidden_state, attention_mask_a)

        if hasattr(transformer_outputs_b, "pooler_output") and transformer_outputs_b.pooler_output is not None:
            pooled_b = transformer_outputs_b.pooler_output
        else:
            pooled_b = self.pooler(transformer_outputs_b.last_hidden_state, attention_mask_b)

        # Combine embeddings per emb_type
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
        else:
            raise ValueError("invalid emb_type")

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
        assert (rubric_mask.sum(dim=1) > 0).all(), "Every sample needs at least one valid rubric."

        scaled_logits = logits / tau
        masked_logits = scaled_logits.masked_fill(rubric_mask == 0, -1e9)

        pos_mask = rubric_mask.gather(1, pos_idx.view(-1, 1)).squeeze(1)
        assert (pos_mask == 1).all(), "pos_idx must refer to valid rubrics."

        return nn.CrossEntropyLoss()(masked_logits, pos_idx)