import logging
from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, SequenceClassifierOutput
from transformers.models.llama.modeling_llama import (
    LlamaPreTrainedModel, 
    LlamaDecoderLayer, 
    LlamaRMSNorm, 
    LlamaRotaryEmbedding,
)
from .modelling_utils import (
    get_noncausal_attention_mask,
    get_noncausal_attention_mask_0,
    get_backward_attention_mask,
    flip_tensor,
    Pooler,
)
from functools import partial
from transformers.utils import TransformersKwargs, logging, auto_docstring
from transformers.processing_utils import Unpack
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.utils.generic import check_model_inputs

logger = logging.get_logger(__name__)
@auto_docstring
class LlamaModel(LlamaPreTrainedModel):
    def __init__(self, config):
        print("Using custom LlamaModel with extended architecture")
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # Initialize standard Llama components
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        # -----------------------------------------------------------------------------------------------------------------
        self.architecture = getattr(config, 'architecture', 'NONE')
        self.mask_type = getattr(config, 'mask_type', "MASK0")
        self.num_unsink_layers = getattr(config, 'num_unsink_layers', 0)
        self.num_bidir_layers = getattr(config, 'num_bidir_layers', 0)
        self.unsink_layers = set(getattr(config, 'unsink_layers', set()))
        self.bidir_layers = set(getattr(config, 'bidir_layers', set()))
        self.connect_layers = getattr(config, 'res_connect', None)
        self.num_hidden_layers = config.num_hidden_layers # the total number of converted backbone layers
        num_converted_layers = self.num_unsink_layers + self.num_bidir_layers
        _is_mask0 = self.mask_type == "MASK0"

        assert not ((self.unsink_layers or self.bidir_layers) and (self.num_unsink_layers or self.num_bidir_layers))
        assert num_converted_layers <= self.num_hidden_layers

        if self.architecture == "EXTEND":
            self.num_hidden_layers += num_converted_layers
            self.layers.extend([LlamaDecoderLayer(config, config.num_hidden_layers + layer_idx) for layer_idx in range(num_converted_layers)])
        elif self.architecture == "EXTRA":
            self.num_hidden_layers = num_converted_layers

        if not (self.unsink_layers or self.bidir_layers):
            self.unsink_layers = self.num_hidden_layers - self.num_unsink_layers
            self.bidir_layers = self.num_hidden_layers - self.num_unsink_layers - self.num_bidir_layers
            for i in range(self.bidir_layers, self.num_hidden_layers if _is_mask0 else self.unsink_layers):
                self.layers[i].self_attn.is_causal = False
        else:
            self.unsink_layers = {layer if layer >= 0 else layer + self.num_hidden_layers for layer in self.unsink_layers}
            self.bidir_layers = {layer if layer >= 0 else layer + self.num_hidden_layers for layer in self.bidir_layers}
            for i in self.bidir_layers | (self.unsink_layers if _is_mask0 else set()):
                self.layers[i].self_attn.is_causal = False
        # -----------------------------------------------------------------------------------------------------------------

        # Initialize weights and apply final processing
        self.post_init()
    @check_model_inputs
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position: torch.Tensor = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )


        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        _is_flash_attn = self.config._attn_implementation == "flash_attention_2"
        _is_intera = self.architecture in {'INTER', 'EXTRA'}
        _is_mask0 = self.mask_type == "MASK0"

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids
        )

        # Get input shape from either input_ids or inputs_embeds
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape[:2]
        
        bidir_attention_mask = get_noncausal_attention_mask(self, attention_mask, input_shape)        
        unsink_attention_mask = get_noncausal_attention_mask_0(self, attention_mask, input_shape) if _is_mask0 else \
                                get_backward_attention_mask(self, attention_mask, inputs_embeds, True)
        
        hidden_states = inputs_embeds
        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        
        # decoder layers
        h1, h2 = None, 0
        for i in range(self.num_hidden_layers):
            decoder_layer = self.layers[i]
            if isinstance(self.unsink_layers, int):
                is_unsink = i >= self.unsink_layers
                is_bidir = i >= self.bidir_layers and not is_unsink
            else:
                is_unsink = i in self.unsink_layers
                is_bidir = i in self.bidir_layers
            layer_mask = unsink_attention_mask if is_unsink else \
                    bidir_attention_mask if is_bidir else causal_mask
            
            reverse_flag = is_unsink and _is_flash_attn and not _is_mask0
            if i == self.bidir_layers:
                h1 = hidden_states


            hidden_states = flip_tensor(hidden_states, reverse_flag)


            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

            hidden_states = flip_tensor(hidden_states, reverse_flag)


        forward_hidden_states = h1 if _is_intera else h2
        for i in range(self.bidir_layers, self.config.num_hidden_layers if _is_intera else 0):
            decoder_layer = self.layers[i]

            tem = decoder_layer.self_attn.is_causal
            decoder_layer.self_attn.is_causal = True

            forward_hidden_states = decoder_layer(
                forward_hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                past_key_value=past_key_values,
                **kwargs,
            )
            
            decoder_layer.self_attn.is_causal = tem

        hidden_states += forward_hidden_states

        hidden_states = self.norm(hidden_states)
        # add hidden states from the last decoder layer



        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,

        )

    
    def freeze_model(self, config=None):
        if config.freeze_type == "all":
            for param in self.parameters():
                param.requires_grad = False

        elif config.freeze_type == "backbone":
            self.embed_tokens.weight.requires_grad = False

            for param in self.layers[:self.num_hidden_layers - config.num_unfreeze_layers].parameters():
                param.requires_grad = False

            if config.num_unfreeze_layers == 0:
                self.norm.weight.requires_grad = False

    def model_init(self):
        first_new_layer = self.config.num_hidden_layers
        for layer in range(first_new_layer, len(self.layers)):
            self.layers[layer].load_state_dict(self.layers[first_new_layer - 1].state_dict())


class LlamaForSequenceClassification(LlamaPreTrainedModel):
    """
    Simple wrapper that adds a classification head to the custom LlamaModel
    """
    def __init__(self, config):
        print("Using custom LlamaForSequenceClassification with LlamaModel as backbone")
        super().__init__(config)
        self.num_labels = config.num_labels
        
        # Use the custom LlamaModel as backbone
        self.model = LlamaModel(config)
        
        # Simple classification head - ensure float32 precision to avoid quantization issues
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.pooler = Pooler(pool_type=getattr(config, 'pool_type', 'avg'))

        # Initialize weights
        self.post_init()
        

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> Union[Tuple, SequenceClassifierOutput]:
        
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Get hidden states
        hidden_states = outputs[0]  # (batch_size, seq_len, hidden_size)

        pooled_output = self.pooler(hidden_states, attention_mask)
        
        # Classification
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # Classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits, labels)
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )
if __name__ == "__main__":
    model= LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-3.1-8B")