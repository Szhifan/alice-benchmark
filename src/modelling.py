import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig, 
    LlamaModel
)
import torch
from torch.nn import CrossEntropyLoss, Bilinear, CosineEmbeddingLoss
from torch.nn.functional import cosine_similarity
from dataclasses import dataclass
from typing import Optional
import re 
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
lora_config = LoraConfig(
        r=128,
        lora_alpha=128,
        lora_dropout=0.1,
        bias="none"
        )
@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
def mean_pooling(
    token_embeddings: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Perform mean pooling on the model output.
    """
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask 

class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, hidden_size: int, n_labels: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.out_proj = nn.Linear(hidden_size, n_labels)
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
class LatentAttention(nn.Module):
    """
    Latent Attention module where the query is the last hidden representation of an LLM,
    and key and value are learnable parameters. 
    """
    def __init__(self, hidden_dim, num_latent_vectors=512, dropout_prob=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_latent_vectors = num_latent_vectors
        
        # Learnable key and value parameters
        self.key = nn.Parameter(torch.randn(num_latent_vectors, hidden_dim))
        self.value = nn.Parameter(torch.randn(num_latent_vectors, hidden_dim))
        
        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Multi-head attention with 1 head
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout_prob,
            batch_first=True
        )
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.key)
        nn.init.xavier_uniform_(self.value)

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: Hidden representation from LLM [batch_size, seq_len, hidden_dim]
        Returns:
            context_vector: Attended output [batch_size, seq_len, hidden_dim]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Use hidden states as queries
        query = hidden_states  # [batch_size, seq_len, hidden_dim]
        
        # Expand key and value for batch processing
        key = self.key.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_latent, hidden_dim]
        value = self.value.unsqueeze(0).expand(batch_size, -1, -1)  # [batch_size, num_latent, hidden_dim]
        
        # Apply multi-head attention
        context, _ = self.attention(query, key, value)  # [batch_size, seq_len, hidden_dim]
        
        # Apply layer normalization and residual connection
        context = self.layer_norm(context + hidden_states)
        context = self.dropout(context)
        
        return context

class AsagConfig(PretrainedConfig):

    def __init__(
        self,
        base_model_name_or_path: str = None,
        n_labels: int = 1,
        use_lora: bool = False,
        use_bidirectional: bool = True,  
        use_latent_attention: bool = False, 
        use_label_weights: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path or getattr(self, "name_or_path", None)
        self.n_labels = n_labels
        self.use_lora = use_lora
        self.use_bidirectional = use_bidirectional
        self.use_latent_attention = use_latent_attention,
        self.use_label_weights = use_label_weights

 

class AsagXNet(PreTrainedModel):
    """
    Encoder-based ASAG model inheriting from PreTrainedModel to leverage save_pretrained and from_pretrained
    The encoder model is a sequence classification model (BERT, RoBERTa, etc.)
    """

    def __init__(self, config: AsagConfig):
        super().__init__(config)
        # load underlying sequence classification model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=config.n_labels
        )
        config.encoder_config = self.model.config
        # Class weights for imbalanced dataset (label 0: 2/3, label 1: 1/3)
        self.label_weights = torch.tensor([0.75, 1.5])  if config.use_label_weights else None
        # optional LoRA wrapping
        if config.use_lora:
            lora_config.task_type = TaskType.SEQ_CLS
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # this initializes weights and ties embeddings if needed
        self.post_init()
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        token_type_ids: Optional[torch.Tensor] = None, 
        labels: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )

        loss = None
        if labels is not None:
            if self.config.n_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(outputs.logits.view(-1), labels.view(-1).float())
            else:
                loss_fct = CrossEntropyLoss(weight=self.label_weights.to(outputs.logits.device))
                loss = loss_fct(outputs.logits.view(-1, self.config.n_labels), labels.view(-1))
                outputs.loss = loss

        return ModelOutput(logits=outputs.logits, loss=outputs.loss)

class AsagXNetLlama(PreTrainedModel):
    """
    Llama-based ASAG model inheriting from PreTrainedModel to leverage save_pretrained and from_pretrained
    The encoder model is a sequence classification model (Llama)
    """
    def __init__(self, config: AsagConfig):
        super().__init__(config)
        self.config = config
        self.use_bidirectional = getattr(config, 'use_bidirectional', True)
        self.use_latent_attention = getattr(config, 'use_latent_attention', False)
        
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
        if config.use_lora:
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        self.post_init()
    def get_last_hidden_state(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Find the positions of the last non-padding tokens
        seq_lengths = attention_mask.sum(dim=1) - 1  # -1 because indices are 0-based
        batch_size = hidden_states.shape[0]
        
        # Gather the hidden states of the last tokens
        last_hidden = hidden_states[torch.arange(batch_size, device=hidden_states.device), seq_lengths]
        return last_hidden
    def create_bidirectional_mask(self, attention_mask: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        device = input_ids.device


        bidirectional_mask = torch.zeros(
            (batch_size, 1, seq_length, seq_length),
            dtype=torch.float32, 
            device=device
        )
        if attention_mask is not None:
            seq_lengths = attention_mask.sum(dim=1)
            for i in range(batch_size):
                valid_len = seq_lengths[i].item()
                bidirectional_mask[i, 0, :valid_len, :valid_len] = 0.0

                min_val = torch.finfo(torch.float32).min
                bidirectional_mask[i, 0, valid_len:, :] = min_val
                bidirectional_mask[i, 0, :, valid_len:] = min_val
        
        return bidirectional_mask
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        labels: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        
        # Set bidirectional attention if needed (non-causal)
        if self.use_bidirectional:
            # Create bidirectional attention mask
            extended_attention_mask = self.create_bidirectional_mask(attention_mask, input_ids)
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
            pool_output = mean_pooling(last_hidden_states, attention_mask)
            logits = self.classifier(pool_output)
        else:
            pool_output = self.get_last_hidden_state(last_hidden_states, attention_mask)
            logits = self.classifier(pool_output)
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.n_labels), labels.view(-1))
        return ModelOutput(logits=logits, loss=loss)

class AsagSNet(PreTrainedModel):
    """
    Sentence Transformer model for ASAG that computes cosine similarity between
    separately encoded student answers and rubrics.
    """
    def __init__(self, config: AsagConfig):
        super().__init__(config)
        # load underlying encoder
        self.encoder = AutoModel.from_pretrained(config.base_model_name_or_path)
        config.encoder_config = self.encoder.config

        # this initializes weights and ties embeddings if needed
        self.post_init()
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Mean pooling or use [CLS] token for embeddings
        embeddings = mean_pooling(outputs.last_hidden_state, attention_mask)
        # embeddings =  outputs.last_hidden_state[:, 0, :]

        return embeddings
        
    def forward(
        self, 
        input_ids_a: torch.Tensor,  # Student answer
        attention_mask_a: torch.Tensor,
        input_ids_b: torch.Tensor,  # Rubric
        attention_mask_b: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        
        # Encode student answer
        embeddings_a = self.encode(input_ids_a, attention_mask_a)
        # Encode rubric
        embeddings_b = self.encode(input_ids_b, attention_mask_b)
        
        # Normalize embeddings for cosine similarity
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
        loss = None

        # h = torch.concat([embeddings_a, embeddings_b], dim=-1)
        # logits = self.classifier(h)
        logits = cosine_similarity(embeddings_a, embeddings_b, dim=-1).unsqueeze(1)
        # Convert label 0 to -1 for cosine embedding loss

        loss_fct = CosineEmbeddingLoss()
        if labels is not None:
            label_id_cosine = labels.clone()
            label_id_cosine[label_id_cosine == 0] = -1
            loss = loss_fct(logits, label_id_cosine)
        return ModelOutput(logits=logits, loss=loss)
class AsagSNetLlama(AsagSNet):
    """
    Llama-based Sentence Transformer model for ASAG that computes cosine similarity between
    separately encoded student answers and rubrics.
    """
    def __init__(self, config: AsagConfig):
        # Initialize PreTrainedModel directly instead of AsagSNet.__init__
        PreTrainedModel.__init__(self, config)
        self.config = config
        self.use_bidirectional = getattr(config, 'use_bidirectional', True)
        self.use_latent_attention = getattr(config, 'use_latent_attention', False)
        
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
        config.encoder_config = self.encoder.config
        
        hidden_size = self.encoder.config.hidden_size
        
        if self.use_latent_attention:
            self.latent_attn = LatentAttention(hidden_size)

        if config.use_lora:
            self.encoder = prepare_model_for_kbit_training(self.encoder)
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()
        
        self.post_init()

    # Inherit create_bidirectional_mask from AsagXNetLlama
    create_bidirectional_mask = AsagXNetLlama.create_bidirectional_mask

    def encode(self, input_ids, attention_mask):
        if self.use_bidirectional:
            extended_attention_mask = self.create_bidirectional_mask(attention_mask, input_ids)
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=extended_attention_mask
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
        
        last_hidden_state = outputs.last_hidden_state
        
        if self.use_latent_attention:
            last_hidden_state = self.latent_attn(last_hidden_state)
        
        embeddings = mean_pooling(last_hidden_state, attention_mask)
        return embeddings
        
if __name__ == "__main__":
    model_config = AsagConfig(base_model_name_or_path="meta-llama/Llama-3.2-1B", n_labels=1, use_lora=False, use_latent_attention=True, use_bidirectional=True)
    asagllm = AsagXNetLlama(model_config)
    input_ids = torch.randint(0, 1000, (2, 128))  # Example input
    attention_mask = torch.randint(0, 2, (2, 128))  # Example attention mask
    outputs = asagllm(input_ids=input_ids, attention_mask=attention_mask)
    print(outputs)