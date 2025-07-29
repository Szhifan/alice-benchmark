import torch.nn as nn 
from torch import Tensor
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModelForSequenceClassification,
)
from transformers.utils.generic import ModelOutput
import torch 
from typing import Optional, Tuple, Union 
from modelling.modelling_berts import AsagConfig

class AsagBsl(PreTrainedModel):
    """
    Asag Baseline Model which predicts score directly. 
    """
    config_class = AsagConfig
    base_model_prefix = "asag_baseline"
    supports_gradient_checkpointing = True

    def __init__(self, config: AsagConfig):
        super().__init__(config)
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path, 
            n_labels=config.n_labels,
        )
        self.config.update(self.model.config.to_dict())
        self.post_init()


    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple[torch.Tensor], ModelOutput]:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        return outputs
    
class Pooler:
    def __init__(self, pool_type, include_prompt=False):
        self.pool_type = pool_type
        self.include_prompt = include_prompt or self.pool_type in ("cls", "last")

    def __call__(
        self, 
        last_hidden_states: Tensor,
        attention_mask: Tensor,
        prompt_length: int = None,
    ) -> Tensor:
        sequence_lengths = attention_mask.sum(dim=1)
        batch_size = last_hidden_states.shape[0]
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        device = last_hidden_states.device
        
        if not self.include_prompt and prompt_length is not None:
            if left_padding:
                prompt_mask = torch.ones_like(attention_mask)
                range_tensor = torch.arange(attention_mask.size(1), 0, -1, device=device).unsqueeze(0)
                prompt_mask = (range_tensor > (sequence_lengths-prompt_length).unsqueeze(1))
                attention_mask[prompt_mask] = 0
            else:
                attention_mask[:, :prompt_length] = 0
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.pool_type == "avg":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.pool_type == "weightedavg":  # position-weighted mean pooling from SGPT (https://arxiv.org/abs/2202.08904)
            attention_mask *= attention_mask.cumsum(dim=1)  # [0,1,1,1,0,0] -> [0,1,2,3,0,0]
            s = torch.sum(last_hidden * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            emb = s / d
        elif self.pool_type == "cls":
            emb = last_hidden[:, 0]
        elif self.pool_type == "last":
            if left_padding:
                emb = last_hidden[:, -1]
            else:
                emb = last_hidden[torch.arange(batch_size, device=device), sequence_lengths-1]
        else:
            raise ValueError(f"pool_type {self.pool_type} not supported")

        return emb