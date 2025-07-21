import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, T5EncoderModel, PreTrainedModel, PretrainedConfig, AutoModelForSequenceClassification, AutoModel, T5ForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
import torch
from torch.nn import CrossEntropyLoss, Bilinear, CosineEmbeddingLoss
from torch.nn.functional import sigmoid, cosine_similarity
from dataclasses import dataclass
from typing import Optional, Tuple
import re 
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
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
def eos_embeddings(hiden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Get the embeddings of the last non-padding token in each sequence.
    """
    eos_pos = attention_mask.sum(dim=1) - 1  # End of sequence positions
    return hiden_states[torch.arange(hiden_states.size(0)), eos_pos]
def get_tokenizer(base_model: str) -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(base_model)
    tok.sep_token = tok.sep_token or tok.eos_token  # Ensure sep_token is set
    return tok


class AsagConfig(PretrainedConfig):

    def __init__(
        self,
        base_model_name_or_path: str = None,
        n_labels: int = 2,
        use_lora: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name_or_path = base_model_name_or_path or getattr(self, "name_or_path", None)
        self.n_labels = n_labels
        self.use_lora = use_lora


class AsagXNet(PreTrainedModel):
    """
    Encoder-based ASAG model inheriting from PreTrainedModel to leverage save_pretrained and from_pretrained
    """

    def __init__(self, config: AsagConfig, label_weights: Optional[torch.Tensor] = None):
        super().__init__(config)
        # load underlying encoder
        self.label_weights = label_weights
        self.encoder = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path)
        config.encoder_config = self.encoder.config
        # optional LoRA wrapping
        if config.use_lora:
            # freeze_model(self.encoder)
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()
        

        # this initializes weights and ties embeddings if needed
        self.post_init()
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        token_type_ids: Optional[torch.Tensor] = None, 
        label_id: Optional[torch.Tensor] = None
    ) -> ModelOutput:
  
        
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=label_id
        )
        return encoder_outputs
class PointerRubricModel(PreTrainedModel):
    """
    The implementation of GRASP
    """
    def __init__(self, config: AsagConfig):
        super().__init__(config)
        config.model_type = PointerRubricModel
        self.encoder = AutoModel.from_pretrained(config.base_model_name_or_path)

        self.pointer_head = nn.Bilinear(self.encoder.config.hidden_size, self.encoder.config.hidden_size, 1)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        rubric_span: torch.Tensor,
        answer_span: torch.Tensor,
        rubric_mask: torch.Tensor,
        label_id: Optional[torch.Tensor] = None
    ) -> ModelOutput:
        """
        Forward method for the Pointer Rubric Model.
        """
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        seq_embeddings = encoder_outputs.last_hidden_state
        B, R = rubric_span.size(0), rubric_span.size(1)
        H = seq_embeddings.size(-1)
        rubric_embeddings = torch.zeros((B, R, H), dtype=seq_embeddings.dtype, device=seq_embeddings.device)
        for i in range(R):
            starts, ends = rubric_span[:, i, 0], rubric_span[:, i, 1]
            emb = torch.stack([
                seq_embeddings[b, starts[b]:ends[b]].mean(dim=0) 
                if rubric_mask[b, i] else torch.zeros(H, dtype=seq_embeddings.dtype, device=seq_embeddings.device)
                for b in range(B)
            ])
            rubric_embeddings[:, i, :] = emb
        ans_starts, ans_ends = answer_span[:, 0], answer_span[:, 1]
        answer_embeddings = torch.stack([
            seq_embeddings[b, ans_starts[b]:ans_ends[b]].mean(dim=0) 
            for b in range(B)
        ])
        answer_embeddings = answer_embeddings.unsqueeze(1).expand(-1, R, -1)  # Expand to match rubric embeddings
        scores = self.pointer_head(rubric_embeddings, answer_embeddings).squeeze(-1)  # Shape: (B, R)
        scores = scores.masked_fill(~rubric_mask, float('-inf'))  # Apply mask to scores
        logits = F.softmax(scores, dim=-1)  # Convert scores to probabilities
        loss = None
        if label_id is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), label_id.view(-1))
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
        # optional LoRA wrapping
        if config.use_lora:
            # freeze_model(self.encoder)
            self.encoder = get_peft_model(self.encoder, lora_config)
            self.encoder.print_trainable_parameters()

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
        label_id: Optional[torch.Tensor] = None
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
        if label_id is not None:
            label_id_cosine = label_id.clone()
            label_id_cosine[label_id_cosine == 0] = -1
            loss = loss_fct(logits, label_id_cosine)
        return ModelOutput(logits=logits, loss=loss)

        
if __name__ == "__main__":
    model_config = AsagConfig(base_model_name_or_path="bert-base-uncased", n_labels=2, use_lora=False)
    asagxnetmodel = AsagXNet(model_config)
    for n, par in asagxnetmodel.named_parameters():
        print(n, par.shape)