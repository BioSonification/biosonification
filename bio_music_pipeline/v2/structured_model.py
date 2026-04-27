"""Generic autoregressive models for harmony and melody generation."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
    return torch.triu(torch.ones((size, size), device=device, dtype=torch.bool), diagonal=1)


def _sample_top_k_top_p(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        threshold = values[..., -1, None]
        logits = torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


class BioConditionedSequenceModel(nn.Module):
    """A compact autoregressive Transformer with bio memory and prefix masking."""

    def __init__(
        self,
        vocab_size: int,
        bio_vector_dim: int,
        max_seq_len: int,
        pad_token_id: int,
        bos_token_id: int,
        eos_token_id: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.15,
        bio_memory_tokens: int = 4,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.bio_memory_tokens = bio_memory_tokens

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.bio_projection = nn.Sequential(
            nn.Linear(bio_vector_dim, d_model * bio_memory_tokens),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * bio_memory_tokens, d_model * bio_memory_tokens),
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def _bio_memory(self, bio_vector: torch.Tensor) -> torch.Tensor:
        memory = self.bio_projection(bio_vector)
        return memory.view(bio_vector.size(0), self.bio_memory_tokens, -1)

    def forward(
        self,
        input_ids: torch.Tensor,
        bio_vector: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        memory = self._bio_memory(bio_vector)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~attention_mask.bool()
        decoded = self.decoder(
            tgt=hidden,
            memory=memory,
            tgt_mask=_causal_mask(seq_len, input_ids.device),
            tgt_key_padding_mask=key_padding_mask,
        )
        decoded = self.final_norm(decoded)
        return self.lm_head(decoded)

    def compute_loss(
        self,
        input_ids: torch.Tensor,
        bio_vector: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        pair_weight: Optional[torch.Tensor] = None,
    ) -> dict:
        logits = self.forward(
            input_ids=input_ids[:, :-1],
            bio_vector=bio_vector,
            attention_mask=attention_mask[:, :-1] if attention_mask is not None else None,
        )
        labels = input_ids[:, 1:]
        token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            ignore_index=self.pad_token_id,
            reduction="none",
        ).view(labels.size())
        valid_mask = (labels != self.pad_token_id).float()
        if loss_mask is not None:
            valid_mask = valid_mask * loss_mask[:, 1:].float()
        token_loss = (token_loss * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
        if pair_weight is not None:
            token_loss = token_loss * pair_weight
        token_loss = token_loss.mean()
        return {"loss": token_loss}

    @torch.no_grad()
    def generate(
        self,
        bio_vector: torch.Tensor,
        prefix_token_ids: Sequence[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        min_new_tokens: int = 0,
        stop_token_ids: Optional[Sequence[int]] = None,
    ) -> torch.Tensor:
        if bio_vector.dim() == 1:
            bio_vector = bio_vector.unsqueeze(0)
        if bio_vector.size(0) != 1:
            raise ValueError("Generation currently supports batch_size=1.")
        stop_ids = set(stop_token_ids or [self.eos_token_id])
        tokens = torch.tensor(list(prefix_token_ids), dtype=torch.long, device=bio_vector.device).unsqueeze(0)
        for step in range(max_new_tokens):
            if tokens.size(1) >= self.max_seq_len:
                break
            logits = self.forward(tokens, bio_vector)[:, -1, :]
            if step < min_new_tokens and self.eos_token_id in stop_ids:
                logits[:, self.eos_token_id] = float("-inf")
            next_token = _sample_top_k_top_p(logits, temperature=temperature, top_k=top_k, top_p=top_p)
            tokens = torch.cat([tokens, next_token], dim=1)
            if step >= min_new_tokens and int(next_token.item()) in stop_ids:
                break
        return tokens.squeeze(0)
