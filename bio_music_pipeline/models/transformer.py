"""
Stage 4: Bio-conditioned autoregressive transformer decoder.

This module implements the core generative model with:
- Gumbel-Softmax differentiable sampling with straight-through estimation
- Temperature annealing
- Auxiliary language model loss with frozen weights
- Composite loss function balancing CE and auxiliary criteria
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:x.size(0), :, :]
        return self.dropout(x)


class GumbelSoftmaxSampler:
    """
    Gumbel-Softmax differentiable sampling with straight-through estimation.
    
    Enables gradient flow through discrete sampling operations.
    """
    
    def __init__(self, temperature_init: float = 1.0, 
                 temperature_min: float = 0.1,
                 decay: float = 0.9999):
        """
        Initialize Gumbel-Softmax sampler.
        
        Args:
            temperature_init: Initial temperature
            temperature_min: Minimum temperature (annealing floor)
            decay: Temperature decay rate per step
        """
        self.temperature = temperature_init
        self.temperature_init = temperature_init
        self.temperature_min = temperature_min
        self.decay = decay
    
    def sample(self, logits: torch.Tensor, hard: bool = True) -> torch.Tensor:
        """
        Sample from categorical distribution using Gumbel-Softmax.
        
        Args:
            logits: Logits of shape (batch_size, vocab_size) or (seq_len, batch, vocab)
            hard: If True, use straight-through estimator
            
        Returns:
            One-hot encoded sample with gradient flow
        """
        # Handle different input shapes
        original_shape = logits.shape
        if logits.dim() == 3:
            logits_flat = logits.view(-1, logits.size(-1))
        else:
            logits_flat = logits
        
        # Sample from Gumbel distribution
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits_flat) + 1e-20) + 1e-20)
        
        # Apply temperature
        y_soft = F.softmax((logits_flat + gumbel_noise) / self.temperature, dim=-1)
        
        if hard:
            # Straight-through estimator
            index = y_soft.max(dim=-1, keepdim=True)[1]
            y_hard = torch.zeros_like(logits_flat).scatter_(-1, index, 1.0)
            # Stop gradient on y_hard, but allow gradient through y_soft
            ret = y_hard - y_soft.detach() + y_soft
        else:
            ret = y_soft
        
        # Reshape back to original shape
        if len(original_shape) == 3:
            ret = ret.view(original_shape)
        
        return ret
    
    def step(self):
        """Anneal temperature."""
        self.temperature = max(self.temperature_min, self.temperature * self.decay)
    
    def reset(self):
        """Reset temperature to initial value."""
        self.temperature = self.temperature_init


class BioConditioningModule(nn.Module):
    """
    Projects bio-vectors into conditioning embeddings for transformer.
    """
    
    def __init__(self, bio_vector_dim: int, d_model: int, 
                 conditioning_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        
        self.bio_projection = nn.Sequential(
            nn.Linear(bio_vector_dim, conditioning_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(conditioning_dim * 2, conditioning_dim),
            nn.LayerNorm(conditioning_dim)
        )
        
        self.conditioning_fusion = nn.Sequential(
            nn.Linear(d_model + conditioning_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)
        )
    
    def forward(self, hidden: torch.Tensor, bio_vector: torch.Tensor) -> torch.Tensor:
        """
        Fuse bio-conditioning with hidden states.
        
        Args:
            hidden: Hidden states of shape (seq_len, batch, d_model)
            bio_vector: Bio-vectors of shape (batch, bio_vector_dim)
            
        Returns:
            Conditioned hidden states
        """
        # Project bio-vector
        bio_emb = self.bio_projection(bio_vector)  # (batch, conditioning_dim)
        
        # Expand to sequence length
        bio_emb_expanded = bio_emb.unsqueeze(0).expand(hidden.size(0), -1, -1)
        
        # Concatenate and fuse
        combined = torch.cat([hidden, bio_emb_expanded], dim=-1)
        conditioned = self.conditioning_fusion(combined)
        
        return conditioned


class AuxiliaryLanguageModel(nn.Module):
    """
    Frozen auxiliary language model for auxiliary loss computation.
    
    This module's weights are frozen during training, but gradients
    flow through it to influence the main model's learning.
    """
    
    def __init__(self, d_model: int, vocab_size: int, n_layers: int = 2):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.LSTM(d_model, d_model, batch_first=True)
            for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Freeze weights after initialization
        self._freeze_weights()
    
    def _freeze_weights(self):
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        Compute auxiliary LM predictions.
        
        Args:
            hidden: Hidden states from main model
            
        Returns:
            Logits over vocabulary
        """
        output = hidden
        for layer in self.layers:
            output, _ = layer(output)
        
        logits = self.output_proj(output)
        return logits


class BioConditionedTransformerDecoder(nn.Module):
    """
    Autoregressive transformer decoder with bio-conditioning.
    
    Uses Gumbel-Softmax for differentiable token sampling and includes
    auxiliary language model loss.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 max_seq_len: int = 512,
                 bio_vector_dim: int = 128,
                 dropout: float = 0.1,
                 gumbel_temp_init: float = 1.0,
                 gumbel_temp_min: float = 0.1,
                 gumbel_decay: float = 0.9999):
        """
        Initialize the bio-conditioned transformer decoder.
        
        Args:
            vocab_size: Size of token vocabulary
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            max_seq_len: Maximum sequence length
            bio_vector_dim: Dimension of bio-feature vectors
            dropout: Dropout probability
            gumbel_temp_init: Initial Gumbel temperature
            gumbel_temp_min: Minimum Gumbel temperature
            gumbel_decay: Temperature decay rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Bio-conditioning
        self.bio_conditioning = BioConditioningModule(bio_vector_dim, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Auxiliary LM (frozen)
        self.auxiliary_lm = AuxiliaryLanguageModel(d_model, vocab_size)
        
        # Gumbel-Softmax sampler
        self.gumbel_sampler = GumbelSoftmaxSampler(gumbel_temp_init, gumbel_temp_min, gumbel_decay)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def encode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        """Embed tokens and add positional encoding."""
        emb = self.token_embedding(tokens) * np.sqrt(self.d_model)
        emb = self.positional_encoding(emb)
        return emb
    
    def forward(self, 
                tokens: torch.Tensor,
                bio_vectors: torch.Tensor,
                return_aux_logits: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            tokens: Input token IDs of shape (seq_len, batch)
            bio_vectors: Bio-conditioning vectors of shape (batch, bio_vector_dim)
            return_aux_logits: Whether to return auxiliary LM logits
            
        Returns:
            Dictionary containing:
                - logits: Main model output logits
                - aux_logits: Auxiliary LM logits (if requested)
                - hidden: Final hidden states
        """
        # Embed tokens
        emb = self.encode_tokens(tokens)
        
        # Generate causal mask
        seq_len = tokens.size(0)
        mask = self.generate_square_subsequent_mask(seq_len, tokens.device)
        
        # Pass through transformer
        hidden = self.transformer_decoder(emb, mask=mask)
        
        # Apply bio-conditioning
        hidden = self.bio_conditioning(hidden, bio_vectors)
        
        # Output logits
        logits = self.output_proj(hidden)
        
        result = {
            'logits': logits,
            'hidden': hidden
        }
        
        if return_aux_logits:
            # Compute auxiliary LM logits (gradients flow through frozen module)
            with torch.no_grad():
                aux_logits = self.auxiliary_lm(hidden)
            result['aux_logits'] = aux_logits
        
        return result
    
    def compute_loss(self, 
                     tokens: torch.Tensor,
                     bio_vectors: torch.Tensor,
                     aux_loss_weight: float = 0.1) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss with cross-entropy and auxiliary LM loss.
        
        Args:
            tokens: Target token IDs of shape (seq_len, batch)
            bio_vectors: Bio-conditioning vectors
            aux_loss_weight: Weight for auxiliary loss
            
        Returns:
            Dictionary containing:
                - total_loss: Combined loss
                - ce_loss: Cross-entropy loss
                - aux_loss: Auxiliary LM loss
        """
        # Shift tokens for autoregressive prediction
        input_tokens = tokens[:-1, :]
        target_tokens = tokens[1:, :]
        
        # Forward pass
        output = self.forward(input_tokens, bio_vectors, return_aux_logits=True)
        logits = output['logits']
        aux_logits = output['aux_logits']
        
        # Flatten for loss computation
        logits_flat = logits.view(-1, self.vocab_size)
        target_flat = target_tokens.reshape(-1)
        
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits_flat, target_flat, ignore_index=-1)
        
        # Auxiliary LM loss (gradients flow through frozen module)
        aux_logits_flat = aux_logits.view(-1, self.vocab_size)
        aux_loss = F.cross_entropy(aux_logits_flat, target_flat, ignore_index=-1)
        
        # Composite loss
        total_loss = ce_loss + aux_loss_weight * aux_loss
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss,
            'aux_loss': aux_loss
        }
    
    @torch.no_grad()
    def generate(self,
                 bio_vectors: torch.Tensor,
                 max_len: int = 512,
                 temperature: float = 1.0,
                 use_gumbel: bool = True,
                 bos_token_id: int = 0,
                 eos_token_id: int = 1) -> torch.Tensor:
        """
        Generate token sequences conditioned on bio-vectors.
        
        Args:
            bio_vectors: Bio-conditioning vectors of shape (batch, bio_vector_dim)
            max_len: Maximum generation length
            temperature: Sampling temperature
            use_gumbel: Whether to use Gumbel-Softmax sampling
            bos_token_id: Beginning of sequence token ID
            eos_token_id: End of sequence token ID
            
        Returns:
            Generated token IDs of shape (batch, seq_len)
        """
        batch_size = bio_vectors.size(0)
        device = bio_vectors.device
        
        # Start with BOS token
        tokens = torch.full((max_len, batch_size), bos_token_id, dtype=torch.long, device=device)
        
        generated_lengths = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for i in range(max_len - 1):
            # Get current sequence
            current_tokens = tokens[:i+1, :]
            
            # Forward pass
            emb = self.encode_tokens(current_tokens)
            mask = self.generate_square_subsequent_mask(i+1, device)
            hidden = self.transformer_decoder(emb, mask=mask)
            hidden = self.bio_conditioning(hidden, bio_vectors)
            logits = self.output_proj(hidden)
            
            # Get logits for next token
            next_token_logits = logits[-1, :, :] / temperature
            
            if use_gumbel and self.training:
                # Use Gumbel-Softmax for differentiable sampling
                sampled = self.gumbel_sampler.sample(next_token_logits, hard=True)
                next_tokens = sampled.argmax(dim=-1)
                self.gumbel_sampler.step()  # Anneal temperature
            else:
                # Standard sampling
                probs = F.softmax(next_token_logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            
            # Update tokens
            tokens[i+1, :] = next_tokens
            
            # Check for EOS
            eos_mask = (next_tokens == eos_token_id)
            newly_finished = eos_mask & (generated_lengths == 0)
            generated_lengths = torch.where(newly_finished, torch.tensor(i+1, device=device), generated_lengths)
        
        return tokens.transpose(0, 1)  # (batch, seq_len)
    
    def update_temperature(self):
        """Update Gumbel temperature (call after each training step)."""
        self.gumbel_sampler.step()


def create_bio_music_model(config: Dict) -> BioConditionedTransformerDecoder:
    """
    Create bio-conditioned transformer model from config dictionary.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized BioConditionedTransformerDecoder
    """
    model = BioConditionedTransformerDecoder(
        vocab_size=config.get('vocab_size', 512),
        d_model=config.get('d_model', 256),
        n_heads=config.get('n_heads', 8),
        n_layers=config.get('n_layers', 6),
        max_seq_len=config.get('max_seq_len', 512),
        bio_vector_dim=config.get('bio_vector_dim', 128),
        dropout=config.get('dropout', 0.1),
        gumbel_temp_init=config.get('gumbel_temp_init', 1.0),
        gumbel_temp_min=config.get('gumbel_temp_min', 0.1),
        gumbel_decay=config.get('gumbel_decay', 0.9999)
    )
    
    return model
