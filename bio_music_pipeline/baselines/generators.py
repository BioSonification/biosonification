"""
Baseline generators for comparison.

Includes:
- Random baseline
- Markov chain baseline
- Unconditional generation
- Rule-based baseline
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class RandomBaseline:
    """Random token generation baseline."""
    
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
    
    def generate(self, n_samples: int, max_len: int, 
                 bos_token: int = 0, eos_token: int = 1) -> torch.Tensor:
        """Generate random sequences."""
        sequences = []
        for _ in range(n_samples):
            seq = [bos_token]
            for _ in range(max_len - 2):
                if np.random.random() < 0.1:  # 10% chance of EOS
                    seq.append(eos_token)
                    break
                token = np.random.randint(2, self.vocab_size)
                seq.append(token)
            if len(seq) < max_len:
                seq.extend([eos_token] * (max_len - len(seq)))
            sequences.append(seq[:max_len])
        return torch.tensor(sequences, dtype=torch.long)


class MarkovBaseline:
    """Markov chain baseline trained on data statistics."""
    
    def __init__(self, vocab_size: int, order: int = 2):
        self.vocab_size = vocab_size
        self.order = order
        self.transition_probs = {}
    
    def fit(self, sequences: List[List[int]]):
        """Fit Markov model to sequences."""
        counts = defaultdict(lambda: defaultdict(int))
        
        for seq in sequences:
            for i in range(len(seq) - self.order):
                context = tuple(seq[i:i+self.order])
                next_token = seq[i+self.order]
                counts[context][next_token] += 1
        
        # Convert to probabilities
        self.transition_probs = {}
        for context, next_counts in counts.items():
            total = sum(next_counts.values())
            self.transition_probs[context] = {
                token: count / total 
                for token, count in next_counts.items()
            }
    
    def generate(self, n_samples: int, max_len: int,
                 bos_token: int = 0, eos_token: int = 1) -> torch.Tensor:
        """Generate sequences using Markov chain."""
        sequences = []
        
        for _ in range(n_samples):
            # Start with BOS tokens
            context = [bos_token] * self.order
            seq = context.copy()
            
            for _ in range(max_len - self.order):
                context_tuple = tuple(context[-self.order:])
                
                # Get transition probabilities
                if context_tuple in self.transition_probs:
                    probs_dict = self.transition_probs[context_tuple]
                    tokens = list(probs_dict.keys())
                    probs = list(probs_dict.values())
                    next_token = np.random.choice(tokens, p=probs)
                else:
                    # Fallback to random
                    next_token = np.random.randint(2, self.vocab_size)
                
                seq.append(next_token)
                context.append(next_token)
                
                # Check for EOS
                if next_token == eos_token:
                    break
            
            # Pad or truncate
            if len(seq) < max_len:
                seq.extend([eos_token] * (max_len - len(seq)))
            sequences.append(seq[:max_len])
        
        return torch.tensor(sequences, dtype=torch.long)


class UnconditionalTransformer(nn.Module):
    """Unconditional transformer decoder (no bio-conditioning)."""
    
    def __init__(self, vocab_size: int, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 6,
                 max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(
            torch.randn(max_seq_len, d_model) * 0.1
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4, dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        emb = self.token_embedding(tokens) + self.positional_encoding[:tokens.size(0)]
        mask = torch.triu(torch.ones(tokens.size(0), tokens.size(0)), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        hidden = self.transformer_decoder(emb, mask=mask)
        return self.output_proj(hidden)
    
    @torch.no_grad()
    def generate(self, n_samples: int, max_len: int,
                 bos_token: int = 0, eos_token: int = 1,
                 temperature: float = 1.0) -> torch.Tensor:
        device = next(self.parameters()).device
        tokens = torch.full((max_len, n_samples), bos_token, 
                           dtype=torch.long, device=device)
        
        for i in range(max_len - 1):
            current = tokens[:i+1, :]
            logits = self.forward(current)
            next_logits = logits[-1, :, :] / temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(probs, 1).squeeze(-1)
            tokens[i+1, :] = next_tokens
        
        return tokens.transpose(0, 1)


class RuleBasedGenerator:
    """Rule-based generation using sonification parameters directly."""
    
    def __init__(self, vocab_size: int, pitch_range: Tuple[int, int] = (36, 96)):
        self.vocab_size = vocab_size
        self.pitch_range = pitch_range
    
    def generate_from_params(self, musical_params: Dict, n_samples: int, 
                            max_len: int) -> torch.Tensor:
        """
        Generate sequences based on musical parameters.
        
        Args:
            musical_params: Dictionary with keys: tempo, key, pitch_range, etc.
            n_samples: Number of sequences to generate
            max_len: Maximum sequence length
            
        Returns:
            Generated sequences
        """
        sequences = []
        
        # Extract parameters
        tempo = musical_params.get('tempo', 120)
        pitch_min = musical_params.get('pitch_min', self.pitch_range[0])
        pitch_max = musical_params.get('pitch_max', self.pitch_range[1])
        rhythm_complexity = musical_params.get('rhythm_complexity', 0.5)
        
        # Map tempo to note density
        note_density = 0.3 + (tempo / 180.0) * 0.4
        
        for _ in range(n_samples):
            seq = [0]  # BOS
            current_pitch = (pitch_min + pitch_max) // 2
            
            for _ in range(max_len - 2):
                if np.random.random() > note_density:
                    # Rest (time shift)
                    shift = np.random.randint(1, min(50, int((1 - rhythm_complexity) * 100)))
                    seq.append(2 + shift)  # SHIFT token
                else:
                    # Note
                    pitch_variance = int((pitch_max - pitch_min) * rhythm_complexity)
                    pitch = current_pitch + np.random.randint(-pitch_variance//2, pitch_variance//2 + 1)
                    pitch = max(pitch_min, min(pitch_max, pitch))
                    
                    velocity = np.random.randint(60, 100)
                    
                    seq.append(2 + 101 + (pitch - self.pitch_range[0]))  # NOTE_ON
                    seq.append(2 + 101 + 176 + velocity)  # VEL
                
                current_pitch = pitch
                
                if np.random.random() < 0.05:
                    seq.append(1)  # EOS
                    break
            
            if len(seq) < max_len:
                seq.extend([1] * (max_len - len(seq)))
            sequences.append(seq[:max_len])
        
        return torch.tensor(sequences, dtype=torch.long)


def create_baselines(config: Dict):
    """Create all baseline generators from config."""
    vocab_size = config.get('vocab_size', 512)
    
    baselines = {
        'random': RandomBaseline(vocab_size),
        'markov': MarkovBaseline(vocab_size, order=2),
        'unconditional': UnconditionalTransformer(
            vocab_size=vocab_size,
            d_model=config.get('d_model', 256),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 6)
        ),
        'rule_based': RuleBasedGenerator(vocab_size)
    }
    
    return baselines
