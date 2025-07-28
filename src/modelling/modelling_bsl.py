import torch.nn as nn 
import torch.nn
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModelForSequenceClassification,
)
from transformers.utils.generic import ModelOutput
import torch 
from typing import Optional, Tuple, Union 
from .modelling import AsagConfig


class AsagBsl(PreTrainedModel):
    """
    Asag Baseline Model which predicts score directly. 
    """
    config_class = AsagConfig
    base_model_prefix = "asag_baseline"
    supports_gradient_checkpointing = True

    def __init__(self, config: PretrainedConfig):
        super().__init__(config)
        self.config = config
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path, 
            num_labels=config.num_labels,
        )


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