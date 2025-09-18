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
        self.score = nn.Linear(input_size, self.num_labels, bias=False)
    def forward(
        self,
        input_ids_a: torch.LongTensor = None,
        input_ids_b: torch.LongTensor = None,
        attention_mask_a: torch.Tensor = None,
        attention_mask_b: torch.Tensor = None,
        labels: Optional[torch.LongTensor] = None,

    ) -> Union[Tuple, SequenceClassifierOutput]:
        
        transformer_outputs_a = self.encoder(
            input_ids=input_ids_a,
            attention_mask=attention_mask_a,
        )

        transformer_outputs_b = self.encoder(
            input_ids=input_ids_b,
            attention_mask=attention_mask_b,

        )
        pooled_a = self.pooler(transformer_outputs_a.last_hidden_state, attention_mask_a)
        pooled_b = self.pooler(transformer_outputs_b.last_hidden_state, attention_mask_b)

        if self.emb_type == 'diff':
            hidden_states = torch.as_tensor(pooled_a - pooled_b)
        elif self.emb_type == 'diffABS':
            hidden_states = torch.abs(torch.as_tensor(pooled_a - pooled_b))
        elif self.emb_type == 'n-diffABS':
            diff = torch.abs(torch.as_tensor(pooled_a - pooled_b))
            hidden_states = torch.cat((pooled_a, diff),1)
        elif self.emb_type == 'n-diffABS-o':
            diff = torch.abs(torch.as_tensor(pooled_a - pooled_b))
            hidden_states = torch.cat((pooled_a, diff, pooled_b),1)
        elif self.emb_type == 'n-o':
            hidden_states = torch.cat((pooled_a, pooled_b),1)

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


        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled_logits
        )