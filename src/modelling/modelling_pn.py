from modelling.modelling_utils import BaseAsagModel
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
from typing import Optional
import math
class PointerRubricGrasp(BaseAsagModel):
    def __init__(self, config, lora_config=None, bnb_config=None):
        super().__init__(config, lora_config=lora_config, bnb_config=bnb_config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)
        self.pointer_head = nn.Bilinear(config.hidden_size, config.hidden_size, 1)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        rubric_spans: torch.Tensor,
        answer_span: torch.Tensor,
        rubric_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        tau: float = 1.0,
    ) -> SequenceClassifierOutput:
        # Encode
        outputs = self.get_encoder_outputs(input_ids, attention_mask)
        token_embeddings = outputs.last_hidden_state  # [B, T, H]

        B, R = rubric_spans.shape[:2]
        H = token_embeddings.shape[-1]

        # Process rubric spans
        rubric_embeddings = []
        for i in range(R):
            starts = rubric_spans[:, i, 0]
            ends = rubric_spans[:, i, 1]
            emb = torch.stack([
                token_embeddings[b, starts[b]:ends[b]].mean(dim=0)
                if rubric_mask[b, i]
                else torch.zeros(H, device=token_embeddings.device, dtype=token_embeddings.dtype)
                for b in range(B)
            ])
            rubric_embeddings.append(emb)
        rubric_embeddings = torch.stack(rubric_embeddings, dim=1)  # [B, R, H]

        # Process answer spans
        a_starts = answer_span[:, 0]
        a_ends = answer_span[:, 1]
        answer_emb = torch.stack([
            token_embeddings[b, a_starts[b]:a_ends[b]].mean(dim=0)
            for b in range(B)
        ])  # [B, H]

        # Compute pointer scores
        answer_exp = answer_emb.unsqueeze(1).expand(-1, R, -1)
        scores = self.pointer_head(rubric_embeddings, answer_exp).squeeze(-1)  # [B, R]

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss = self.listwise_loss(scores, rubric_mask, labels, tau=tau)

        return SequenceClassifierOutput(loss=loss, logits=scores)


class PointerRubricTolegra(BaseAsagModel):
    """
    Tolegra model using attention-based token alignment between answer and rubrics.
    Similar to GRASP but with sophisticated token-level attention alignment.
    """
    
    def __init__(self, config, lora_config=None, bnb_config=None):
        super().__init__(config, lora_config=lora_config, bnb_config=bnb_config)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob if hasattr(config, 'hidden_dropout_prob') else 0.1)
        
        # Projection layers for answer and rubric tokens
        self.answer_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.rubric_proj = nn.Linear(config.hidden_size, config.hidden_size)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        rubric_spans: torch.Tensor,
        answer_span: torch.Tensor,
        rubric_mask: torch.Tensor,
        labels: Optional[torch.LongTensor] = None,
        tau: float = 1.0,
        return_alignments: bool = False,
    ) -> SequenceClassifierOutput:
        """
        Forward pass with attention-based alignment.
        
        Args:
            input_ids: [B, T] token ids
            attention_mask: [B, T] attention mask
            rubric_spans: [B, R, 2] start/end positions of rubrics
            answer_span: [B, 2] start/end positions of answers
            rubric_mask: [B, R] mask for valid rubrics
            labels: [B] ground truth labels
            tau: temperature for loss
            return_alignments: whether to return attention alignments
        """
        # Encode
        outputs = self.get_encoder_outputs(input_ids, attention_mask)
        token_embeddings = outputs.last_hidden_state  # [B, T, H]

        B, R = rubric_spans.shape[:2]
        H = token_embeddings.shape[-1]

        # Get answer spans
        answer_starts = answer_span[:, 0]
        answer_ends = answer_span[:, 1]

        all_alignments = []
        scores_per_rubric = []

        # Process each rubric position
        for i in range(R):
            rubric_starts = rubric_spans[:, i, 0]
            rubric_ends = rubric_spans[:, i, 1]

            rubric_scores = []
            rubric_alignments = []

            # Process each sample in batch
            for b in range(B):
                if not rubric_mask[b, i]:
                    # Invalid rubric - assign negative infinity score
                    rubric_scores.append(torch.tensor(-float("inf"), device=token_embeddings.device))
                    if return_alignments:
                        rubric_alignments.append(torch.zeros(
                            (answer_ends[b] - answer_starts[b], rubric_ends[b] - rubric_starts[b]),
                            device=token_embeddings.device
                        ))
                    continue

                # Extract token embeddings for answer and rubric spans
                answer_tokens = token_embeddings[b, answer_starts[b]:answer_ends[b]]  # [ans_len, H]
                rubric_tokens = token_embeddings[b, rubric_starts[b]:rubric_ends[b]]  # [rub_len, H]

                # Apply projections with dropout
                proj_answer = self.dropout(self.answer_proj(answer_tokens))  # [ans_len, H]
                proj_rubric = self.dropout(self.rubric_proj(rubric_tokens))   # [rub_len, H]

                # Compute token-token alignment scores
                alignment = torch.matmul(proj_answer, proj_rubric.T) / math.sqrt(H)  # [ans_len, rub_len]
                
                if return_alignments:
                    rubric_alignments.append(alignment)

                # Attention-style aggregation over rubric tokens
                # Softmax over rubric dimension to get attention weights
                weights = torch.softmax(alignment, dim=1)  # [ans_len, rub_len]
                
                # Weighted alignment scores: sum over rubric tokens, mean over answer tokens
                rubric_score = (weights * alignment).sum(dim=1).mean()  # scalar
                
                rubric_scores.append(rubric_score)

            # Stack scores for this rubric position
            rubric_scores = torch.stack(rubric_scores)  # [B]
            scores_per_rubric.append(rubric_scores)

            if return_alignments:
                all_alignments.append(rubric_alignments)

        # Final logits: [B, R]
        scores = torch.stack(scores_per_rubric, dim=1)  # [B, R]
        
        # Mask invalid rubrics
        scores = scores.masked_fill(~rubric_mask, float("-inf"))

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = self.listwise_loss(scores, rubric_mask, labels, tau=tau)

        result = SequenceClassifierOutput(loss=loss, logits=scores)
        
        if return_alignments:
            result.alignments = all_alignments
            
        return result