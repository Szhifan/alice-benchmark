import torch
import torch.nn as nn
import warnings
from torch import Tensor
from transformers import PretrainedConfig
class AsagConfig(PretrainedConfig):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)
def create_bidirectional_mask(attention_mask: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    batch_size, seq_length = input_ids.shape
    device = input_ids.device
    dtype = input_ids.dtype

    bidirectional_mask = torch.zeros(
        (batch_size, 1, seq_length, seq_length),
        dtype=dtype, 
        device=device
    )
    if attention_mask is not None:
        seq_lengths = attention_mask.sum(dim=1)
        for i in range(batch_size):
            valid_len = seq_lengths[i].item()
            bidirectional_mask[i, 0, :valid_len, :valid_len] = 0.0

            min_val = torch.finfo(dtype).min
            bidirectional_mask[i, 0, valid_len:, :] = min_val
            bidirectional_mask[i, 0, :, valid_len:] = min_val
    
    return bidirectional_mask

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
    whereas key and value are learnable parameters.
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