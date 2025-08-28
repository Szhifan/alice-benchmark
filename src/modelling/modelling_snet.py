# Built upon the huggingface implementation 

from typing import List, Optional, Tuple, Union
from transformers import AutoModel
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import SequenceClassifierOutput
from .modelling_utils import Pooler
from transformers.utils import logging
from peft import PeftModel, get_peft_model
logger = logging.get_logger(__name__)
_CONFIG_FOR_DOC = "LlamaConfig"


class AsagSNet(nn.Module):
    def __init__(self, config, lora_config=None, emb_type=None, bnb_config=None):
        super().__init__()
        self.config = config
        self.num_labels = config.num_labels
        self.emb_type = emb_type if emb_type is not None else 'diffABS'
        self.pooler = Pooler(pool_type=config.pool_type)
        config.emb_type = self.emb_type
        self.lora_config = lora_config
        if bnb_config is not None:
            self.encoder = AutoModel.from_pretrained(config.base_model_name_or_path, bnb_config=bnb_config)
        else:
            self.encoder = AutoModel.from_pretrained(config.base_model_name_or_path)


        if self.emb_type in['diff','diffABS']:
            input_size = config.hidden_size
        elif self.emb_type in ['n-o','n-diffABS']:
            input_size = config.hidden_size*2
        elif self.emb_type in ['n-diffABS-o']:
            input_size = config.hidden_size*3
        else:
            raise ValueError("invalid emb_type")
        self.score = nn.Linear(input_size, self.num_labels, bias=False)

     
        # Initialize weights and apply final processing

    def init_peft(self):
        self.lora_config.TASK_TYPE = None
        self.encoder = get_peft_model(self.encoder, self.lora_config)
    def load_peft_model(self,cp_path):
        self.encoder = PeftModel.from_pretrained(self.encoder, cp_path)
    def forward(
        self,
        input_ids_a: torch.LongTensor = None,
        input_ids_b: torch.LongTensor = None,
        attention_mask_a: torch.Tensor = None,
        attention_mask_b: torch.Tensor = None,
        labels: Optional[torch.LongTensor] = None,

    ) -> Union[Tuple, SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
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
    @classmethod
    def from_pretrained(cls, model_path, config=None, lora_config=None, emb_type=None):

        import os
        import json
        

        if config is None:
            config_path = os.path.join(model_path, 'config.json')
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(config_path)
        

        if emb_type is None:
            emb_type = getattr(config, 'emb_type', 'diffABS')
        
        model = cls(config, lora_config, emb_type)
        adapter_config_path = os.path.join(model_path, 'adapter_config.json')
        if os.path.exists(adapter_config_path):
            model.encoder = PeftModel.from_pretrained(model.encoder, model_path)

            non_peft_params_path = os.path.join(model_path, 'non_peft_params.bin')
            if os.path.exists(non_peft_params_path):
                non_peft_params = torch.load(non_peft_params_path, map_location='cpu')
                current_state_dict = model.state_dict()
                for key, value in non_peft_params.items():
                    if key in current_state_dict:
                        current_state_dict[key] = value
                model.load_state_dict(current_state_dict, strict=False)
        else:
            state_dict_path = os.path.join(model_path, 'pytorch_model.bin')
            if os.path.exists(state_dict_path):
                state_dict = torch.load(state_dict_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
        
        return model

    def save_pretrained(self, save_path):

        import os
        os.makedirs(save_path, exist_ok=True)

        if hasattr(self.encoder, 'save_pretrained') and isinstance(self.encoder, PeftModel):
            self.encoder.save_pretrained(save_path)
            full_state_dict = self.state_dict()
            encoder_keys = [k for k in full_state_dict.keys() if k.startswith('encoder.')]
            for key in encoder_keys:
                full_state_dict.pop(key, None)
            torch.save(full_state_dict, os.path.join(save_path, 'non_peft_params.bin'))
        else:

            torch.save(self.state_dict(), os.path.join(save_path, 'pytorch_model.bin'))
        if hasattr(self, 'config'):
            self.config.save_pretrained(save_path)

