
from transformers import PretrainedConfig
from dataclasses import dataclass, field, asdict
from typing import Optional
import warnings
import torch
from torch import nn
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers import AutoModel, BitsAndBytesConfig
import re
import os
from torch import Tensor
import logging

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

DECODER_MODEL_TYPES = tuple(['gpt2', 'llama', 'mistral', 'qwen2', 'phi3', 'olmo'])
ARCHITECTURES = tuple(['NONE', 'INPLACE', 'EXTEND', 'INTER', 'EXTRA'])

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
@dataclass
class BackwardSupportedArguments:
    architecture: str = field(
        default="None",
        metadata={"help": "Type of architecture to use. Options: " + ", ".join(ARCHITECTURES)}
    )
    mask_type: str = field(
        default="MASK0", 
        metadata={"help": "Type of sink mask to use. Options: MASK0, BACK"}
    )
    num_unsink_layers: int = field(
        default=0, 
        metadata={"help": "Number of layers to change to unsink attention."}
    )
    num_bidir_layers: int = field(
        default=0,
        metadata={"help": "Number of layers to change to bidirectional attention."}
    )
    unsink_layers: Optional[str] = field(
        default=None, 
        metadata={"help": "Manually set specific layers to change to unsink attention."}
    )
    bidir_layers: Optional[str] = field(
        default=None, 
        metadata={"help": "Manually set specific layers to change to bidirectional attention."}
    )
    freeze_type: Optional[str] = field(
        default=None,
        metadata={"help": "Options:\
            all: freeze all parameters in the model.\
            backbone: freeze the backbone of the model, only train output related parameters(unfreeze_layer + norm).\
            default: freeze the backbone of the model, excluding layers changed to unsink or noncausal attention. \
            false\\none: no to freeze the model."
        }
    )
    pool_type: str = field(
        default="last", 
        metadata={"help": "Pooling type for the model output. Options: avg, weightedavg, cls, last"}
    )
    num_prune_layers: Optional[str] = field(
        default=0,
        metadata={"help": "Delete the top n layers of the model."}
    )
    num_fuse_layers: Optional[str] = field(
        default=0,
        metadata={"help": "Fuse the top n layers of the model into one embedding layer."}
    )
    fuse_type: Optional[str] = field(
        default="avg",
        metadata={"help": "Pooling type for the fused layer. Options: avg, weighted"}
    )
    def __post_init__(self):
        self.architecture = self.architecture.upper()
        if self.architecture not in ARCHITECTURES:
            raise ValueError("Invalid Model Architecture. Options: " + ", ".join(ARCHITECTURES))

        self.mask_type = self.mask_type.upper()
        if self.pool_type not in ("avg", "weightedavg", "cls", "last"):
            self.pool_type = "avg"
            print(f"Invalid pool_type {self.pool_type}, using default 'avg' pooling.")
        assert self.mask_type in ("MASK0", "BACK"), "mask_type must be one of ('MASK0', 'BACK')"
        assert self.fuse_type in ("avg", "weighted"), "fuse_type must be one of ('avg', 'weighted')"
    def to_dict(self):
        return asdict(self)


def get_noncausal_attention_mask(self, attention_mask, input_shape, device=None, dtype=None):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    
    if dtype is None:
        dtype = self.dtype

    if not (attention_mask.dim() == 2 and self.config.is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min
    return extended_attention_mask

def get_noncausal_attention_mask_0(self, attention_mask, input_shape, device=None, dtype=None):
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    assert attention_mask[:, 0].sum() == attention_mask.shape[0]
    assert self.config._attn_implementation != "flash_attention_2"
    # attention_mask[:, 0] = 0
    
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask
        return None
    
    if dtype is None:
        dtype = self.dtype

    if not (attention_mask.dim() == 2 and self.config.is_decoder):
        # show warning only if it won't be shown in `create_extended_attention_mask_for_decoder`
        if device is not None:
            warnings.warn(
                "The `device` argument is deprecated and will be removed in v5 of Transformers.", FutureWarning
            )
    # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
    # ourselves in which case we just need to make it broadcastable to all heads.
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )
    
    extended_attention_mask = extended_attention_mask.repeat(1, 1, extended_attention_mask.shape[-1], 1)
    extended_attention_mask[:, :, 1:, 0] = 0

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(dtype).min

    return extended_attention_mask

def get_backward_attention_mask(
    self,
    attention_mask: torch.Tensor,
    input_tensor: torch.Tensor,
    output_attentions: bool,
):
    if self.config._attn_implementation == "flash_attention_2":
        if attention_mask is not None and 0.0 in attention_mask:
            return attention_mask.flip(dims=(-1,))
        return None
    
    dtype, device = input_tensor.dtype, input_tensor.device
    min_dtype = torch.finfo(dtype).min
    sequence_length = input_tensor.shape[1]
    target_length = attention_mask.shape[-1]

    causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)

    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)

    causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
    causal_mask = causal_mask.flip(dims=(-2,-1))

    if attention_mask is not None:
        causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        padding_mask = padding_mask == 0
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask, min_dtype
        )

    if (
        self.config._attn_implementation == "sdpa"
        and attention_mask is not None
        and attention_mask.device.type == "cuda"
        and not output_attentions
    ):
        # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
        # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
        # Details: https://github.com/pytorch/pytorch/issues/110213
        causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)
    return causal_mask

def flip_tensor(tensor, flag=True):
    if flag:
        return tensor.flip(dims=(1,))
    else:
        return tensor
    
