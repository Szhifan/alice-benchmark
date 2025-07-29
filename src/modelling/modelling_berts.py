import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoModel,
    AutoModelForSequenceClassification,
)
from transformers.utils.generic import ModelOutput
import torch
from torch.nn import CrossEntropyLoss, CosineEmbeddingLoss
from torch.nn.functional import cosine_similarity
from typing import Optional
from modelling.modelling_utils import AsagConfig, Pooler


class AsagXNet(PreTrainedModel):
    """
    Encoder-based ASAG model inheriting from PreTrainedModel to leverage save_pretrained and from_pretrained
    The encoder model is a sequence classification model (BERT, RoBERTa, etc.)
    """
    config_class = AsagConfig
    def __init__(self, config: AsagConfig):
        super().__init__(config)
        # load underlying sequence classification model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config.base_model_name_or_path,
            num_labels=config.n_labels
        )
        self.config.update(self.model.config.to_dict())
        # Class weights for imbalanced dataset (label 0: 2/3, label 1: 1/3)
        self.label_weights = torch.tensor([0.75, 1.5])  if config.use_label_weights else None


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
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(outputs.logits.view(-1, self.config.n_labels), labels.view(-1))
                outputs.loss = loss

        return ModelOutput(logits=outputs.logits, loss=outputs.loss)


class AsagSNet(PreTrainedModel):
    """
    Sentence Transformer model for ASAG that computes cosine similarity between
    separately encoded student answers and rubrics.
    """
    config_class = AsagConfig
    def __init__(self, config: AsagConfig):
        super().__init__(config)
        # load underlying encoder
        self.encoder = AutoModel.from_pretrained(config.base_model_name_or_path)
        self.config.update(self.encoder.config.to_dict())

        # this initializes weights and ties embeddings if needed
        self.post_init()
        self.pooler = Pooler(pool_type="avg")
    def encode(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Mean pooling
        embeddings = self.pooler(outputs.last_hidden_state, attention_mask)

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

        
if __name__ == "__main__":
    config = AsagConfig(
        base_model_name_or_path="bert-base-uncased",
        n_labels=5,
        use_bidirectional=True,
        use_latent_attention=False,
        use_label_weights=True,
    )
    model = AsagXNet(config)
    model.save_pretrained("asag_xnet_model")
    model = AsagXNet.from_pretrained("asag_xnet_model")