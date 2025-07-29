import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    BitsAndBytesConfig, 
    LlamaModel
)
from transformers.utils.generic import ModelOutput
import torch
from torch.nn import CrossEntropyLoss
from typing import Optional
from modelling.modelling_berts import AsagSNet
from modelling.modelling_utils import ClassificationHead, create_bidirectional_mask, LatentAttention, Pooler, AsagConfig

class AsagXNetLlama(PreTrainedModel):
    """
    Llama-based ASAG model inheriting from PreTrainedModel to leverage save_pretrained and from_pretrained
    The encoder model is a sequence classification model (Llama)
    """
    config_class = AsagConfig
    def __init__(self, config: AsagConfig):
        super().__init__(config)
        self.config = config
        self.use_bidirectional = getattr(config, 'use_bidirectional', False)
        self.use_latent_attention = getattr(config, 'use_latent_attention', False)
        self.pool_type = getattr(config, 'pool_type', 'avg')
        self.pooler = Pooler(pool_type=self.pool_type)
        
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit = True, # Activate 4-bit precision base model loading
            bnb_4bit_use_double_quant = True, # Activate nested quantization for 4-bit base models (double quantization)
            bnb_4bit_quant_type = "nf4",# Quantization type (fp4 or nf4)
            bnb_4bit_compute_dtype = torch.bfloat16, # Compute data type for 4-bit base models
        )
        
        self.model = LlamaModel.from_pretrained(
            config.base_model_name_or_path,
            quantization_config=self.bnb_config if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        hidden_size = self.model.config.hidden_size
        
        if self.use_latent_attention:
            self.latent_attn = LatentAttention(hidden_size)

        self.classifier = ClassificationHead(hidden_size, config.n_labels)
        self.config.update(self.model.config.to_dict())
        self.post_init()
    def get_last_hidden_state(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Find the positions of the last non-padding tokens
        seq_lengths = attention_mask.sum(dim=1) - 1  # -1 because indices are 0-based
        batch_size = hidden_states.shape[0]
        
        # Gather the hidden states of the last tokens
        last_hidden = hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lengths]
        return last_hidden

    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        
        # Set bidirectional attention if needed (non-causal)
        if self.use_bidirectional:
            # Create bidirectional attention mask
            extended_attention_mask = create_bidirectional_mask(attention_mask, input_ids)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=extended_attention_mask
            )
        else:
            # Use default causal attention
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        # Get the last token representation for classification
    
        last_hidden_states = outputs.last_hidden_state
        if self.use_latent_attention:
            last_hidden_states = self.latent_attn(last_hidden_states)
        out_emb = self.get_last_hidden_state(last_hidden_states, attention_mask)
        logits = self.classifier(out_emb)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.n_labels), labels.view(-1))
        return ModelOutput(logits=logits, loss=loss)


class AsagSNetLlama(AsagSNet):
    """
    Llama-based Sentence Transformer model for ASAG that computes cosine similarity between
    separately encoded student answers and rubrics.
    """
    config_class = AsagConfig
    def __init__(self, config: AsagConfig):
        # Initialize PreTrainedModel directly instead of AsagSNet.__init__
        PreTrainedModel.__init__(self, config)
        self.config = config
        self.use_bidirectional = getattr(config, 'use_bidirectional', True)
        self.use_latent_attention = getattr(config, 'use_latent_attention', False)
        self.pool_type = getattr(config, 'pool_type', 'avg')
        self.pooler = Pooler(pool_type=self.pool_type)
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit = True,
            bnb_4bit_use_double_quant = True,
            bnb_4bit_quant_type = "nf4",
            bnb_4bit_compute_dtype = torch.bfloat16,
        )
        
        # Use Llama encoder instead of AutoModel
        self.encoder = LlamaModel.from_pretrained(
            config.base_model_name_or_path,
            quantization_config=self.bnb_config if torch.cuda.is_available() else None,
            trust_remote_code=True
        )
        
        hidden_size = self.encoder.config.hidden_size
        
        if self.use_latent_attention:
            self.latent_attn = LatentAttention(hidden_size)
        self.config.update(self.encoder.config.to_dict())
        self.post_init()

    def encode(self, input_ids, attention_mask):
        attention_mask = create_bidirectional_mask(attention_mask, input_ids) if self.use_bidirectional else attention_mask
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        if self.use_latent_attention:
            last_hidden_state = self.latent_attn(last_hidden_state)
        out_emb = self.pooler(last_hidden_state, attention_mask)
        return out_emb