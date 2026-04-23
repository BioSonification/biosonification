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
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def _extract_token_groups(vocab: Dict[str, int]) -> Dict[str, object]:
    """Extract token groups and ids from preprocessor vocabulary."""
    bos_id = vocab.get("BOS", 0)
    eos_id = vocab.get("EOS", 1)
    pad_id = vocab.get("PAD", 2)

    shift_ids = sorted(
        [idx for token, idx in vocab.items() if token.startswith("SHIFT_")],
        key=lambda x: x
    )
    note_on_pairs = []
    note_off_pairs = []
    vel_ids = []

    for token, idx in vocab.items():
        if token.startswith("NOTE_ON_"):
            note_on_pairs.append((int(token.split("_")[2]), idx))
        elif token.startswith("NOTE_OFF_"):
            note_off_pairs.append((int(token.split("_")[2]), idx))
        elif token.startswith("VEL_"):
            vel_ids.append(idx)

    note_on_pairs.sort(key=lambda x: x[0])
    note_off_pairs.sort(key=lambda x: x[0])
    vel_ids.sort()

    note_on_by_pitch = {pitch: idx for pitch, idx in note_on_pairs}
    note_off_by_pitch = {pitch: idx for pitch, idx in note_off_pairs}
    all_special = {bos_id, eos_id, pad_id}
    content_ids = sorted([idx for idx in vocab.values() if idx not in all_special])

    return {
        "bos_id": bos_id,
        "eos_id": eos_id,
        "pad_id": pad_id,
        "shift_ids": shift_ids,
        "note_on_by_pitch": note_on_by_pitch,
        "note_off_by_pitch": note_off_by_pitch,
        "velocity_ids": vel_ids,
        "content_ids": content_ids,
    }


class RandomBaseline:
    """Random token generation baseline."""

    def __init__(self, token_groups: Dict[str, object], seed: int = 42):
        self.token_groups = token_groups
        self.seed = np.random.RandomState(seed)

    def generate(self, n_samples: int, max_len: int, min_len: int = 32) -> torch.Tensor:
        """Generate random sequences."""
        bos_id = self.token_groups["bos_id"]
        eos_id = self.token_groups["eos_id"]
        pad_id = self.token_groups["pad_id"]
        content_ids = self.token_groups["content_ids"]

        min_len = int(max(2, min(min_len, max_len - 1)))
        sequences = []
        for _ in range(n_samples):
            seq = [bos_id]
            for step in range(max_len - 2):
                if step + 1 >= min_len and self.seed.rand() < 0.1:
                    seq.append(eos_id)
                    break
                token = int(self.seed.choice(content_ids))
                seq.append(token)
            if len(seq) < max_len:
                seq.extend([pad_id] * (max_len - len(seq)))
            sequences.append(seq[:max_len])
        return torch.tensor(sequences, dtype=torch.long)


class MarkovBaseline:
    """Markov chain baseline trained on data statistics."""

    def __init__(self, token_groups: Dict[str, object], order: int = 2, seed: int = 42):
        self.order = order
        self.token_groups = token_groups
        self.transition_probs = {}
        self.seed = np.random.RandomState(seed)

    def fit(self, sequences: List[List[int]]):
        """Fit Markov model to sequences."""
        counts = defaultdict(lambda: defaultdict(int))

        for seq in sequences:
            for i in range(len(seq) - self.order):
                context = tuple(seq[i:i + self.order])
                next_token = seq[i + self.order]
                counts[context][next_token] += 1

        self.transition_probs = {}
        for context, next_counts in counts.items():
            total = sum(next_counts.values())
            self.transition_probs[context] = {
                token: count / total for token, count in next_counts.items()
            }

    def generate(self, n_samples: int, max_len: int, min_len: int = 32) -> torch.Tensor:
        """Generate sequences using Markov chain."""
        bos_id = self.token_groups["bos_id"]
        eos_id = self.token_groups["eos_id"]
        pad_id = self.token_groups["pad_id"]
        content_ids = self.token_groups["content_ids"]

        min_len = int(max(2, min(min_len, max_len - 1)))
        sequences = []
        for _ in range(n_samples):
            context = [bos_id] * self.order
            seq = context.copy()

            for step in range(max_len - self.order):
                context_tuple = tuple(context[-self.order:])
                if context_tuple in self.transition_probs:
                    probs_dict = self.transition_probs[context_tuple]
                    tokens = list(probs_dict.keys())
                    probs = list(probs_dict.values())
                    next_token = int(self.seed.choice(tokens, p=probs))
                else:
                    next_token = int(self.seed.choice(content_ids))

                seq.append(next_token)
                context.append(next_token)
                if step + self.order >= min_len and next_token == eos_id:
                    break

            if len(seq) < max_len:
                seq.extend([pad_id] * (max_len - len(seq)))
            sequences.append(seq[:max_len])

        return torch.tensor(sequences, dtype=torch.long)


class UnconditionalTransformer(nn.Module):
    """Unconditional transformer decoder baseline."""

    def __init__(self, vocab_size: int, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 6,
                 max_seq_len: int = 512, dropout: float = 0.1,
                 bos_token_id: int = 0, eos_token_id: int = 1, pad_token_id: int = 2):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(max_seq_len, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.size(0)
        emb = self.token_embedding(tokens) * np.sqrt(self.d_model)
        emb = emb + self.positional_encoding[:seq_len]
        hidden = self.transformer(emb, mask=self._causal_mask(seq_len, tokens.device))
        return self.output_proj(hidden)

    def compute_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        input_tokens = tokens[:-1, :]
        targets = tokens[1:, :]
        logits = self.forward(input_tokens)
        return F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            targets.reshape(-1),
            ignore_index=self.pad_token_id,
        )

    @torch.no_grad()
    def generate(self, n_samples: int, max_len: int, temperature: float = 1.0, min_len: int = 32) -> torch.Tensor:
        device = next(self.parameters()).device
        min_len = int(max(2, min(min_len, max_len - 1)))
        tokens = torch.full((max_len, n_samples), self.pad_token_id, dtype=torch.long, device=device)
        tokens[0, :] = self.bos_token_id

        for i in range(max_len - 1):
            current = tokens[:i + 1, :]
            logits = self.forward(current)
            next_logits = logits[-1, :, :] / temperature
            if i + 1 < min_len:
                next_logits[:, self.eos_token_id] = -1e9
            probs = torch.softmax(next_logits, dim=-1)
            next_tokens = torch.multinomial(probs, 1).squeeze(-1)
            tokens[i + 1, :] = next_tokens

        return tokens.transpose(0, 1)


class RuleBasedGenerator:
    """Rule-based generation using sonification parameters directly."""

    def __init__(self, token_groups: Dict[str, object], pitch_range: Tuple[int, int] = (36, 96), seed: int = 42):
        self.token_groups = token_groups
        self.pitch_range = pitch_range
        self.seed = np.random.RandomState(seed)

    def generate_from_params(self, musical_params: Dict, n_samples: int, max_len: int, min_len: int = 32) -> torch.Tensor:
        """Generate sequences from explicit musical parameters."""
        bos_id = self.token_groups["bos_id"]
        eos_id = self.token_groups["eos_id"]
        pad_id = self.token_groups["pad_id"]
        shift_ids = self.token_groups["shift_ids"]
        note_on_by_pitch = self.token_groups["note_on_by_pitch"]
        note_off_by_pitch = self.token_groups["note_off_by_pitch"]
        velocity_ids = self.token_groups["velocity_ids"]

        if not note_on_by_pitch:
            # Fall back to random baseline behavior if note token map is unavailable.
            random_baseline = RandomBaseline(self.token_groups)
            return random_baseline.generate(n_samples, max_len)

        min_len = int(max(2, min(min_len, max_len - 1)))
        tempo = float(musical_params.get('tempo', 120.0))
        rhythm_complexity = float(musical_params.get('rhythm_complexity', 0.5))

        pitch_min, pitch_max = musical_params.get('pitch_range', self.pitch_range)
        pitch_min = int(max(min(note_on_by_pitch.keys()), pitch_min))
        pitch_max = int(min(max(note_on_by_pitch.keys()), pitch_max))

        note_density = np.clip(0.25 + (tempo / 180.0) * 0.45, 0.15, 0.85)

        sequences = []
        for _ in range(n_samples):
            seq = [bos_id]
            current_pitch = int((pitch_min + pitch_max) / 2)

            for step in range(max_len - 2):
                if self.seed.rand() > note_density:
                    if shift_ids:
                        seq.append(int(self.seed.choice(shift_ids)))
                else:
                    pitch_variance = max(1, int((pitch_max - pitch_min) * rhythm_complexity))
                    pitch = int(np.clip(
                        current_pitch + self.seed.randint(-pitch_variance // 2, pitch_variance // 2 + 1),
                        pitch_min,
                        pitch_max,
                    ))

                    if pitch in note_on_by_pitch:
                        seq.append(note_on_by_pitch[pitch])
                        if velocity_ids:
                            seq.append(int(self.seed.choice(velocity_ids)))
                        if pitch in note_off_by_pitch:
                            seq.append(note_off_by_pitch[pitch])
                    current_pitch = pitch

                if step + 1 >= min_len and self.seed.rand() < 0.05:
                    seq.append(eos_id)
                    break

            if len(seq) < max_len:
                seq.extend([pad_id] * (max_len - len(seq)))
            sequences.append(seq[:max_len])

        return torch.tensor(sequences, dtype=torch.long)


class RandomVectorControl:
    """
    Control baseline using random vectors instead of real bio-vectors.

    This tests whether real bio-vectors carry meaningful information
    for music generation, compared to random vectors of the same dimension.
    """

    def __init__(self, vocab_size: int, bio_vector_dim: int = 128, seed: int = 42):
        self.vocab_size = vocab_size
        self.bio_vector_dim = bio_vector_dim
        self.seed = seed
        np.random.seed(seed)

    def generate_random_bio_vectors(self, n_samples: int,
                                     distribution: str = 'gaussian') -> np.ndarray:
        """
        Generate random vectors matching bio-vector statistics.

        Args:
            n_samples: Number of random vectors
            distribution: 'gaussian', 'uniform', or 'matched' (match real stats)

        Returns:
            Array of shape (n_samples, bio_vector_dim)
        """
        if distribution == 'gaussian':
            return np.random.randn(n_samples, self.bio_vector_dim)
        elif distribution == 'uniform':
            return np.random.uniform(-1, 1, (n_samples, self.bio_vector_dim))
        elif distribution == 'matched':
            return np.random.randn(n_samples, self.bio_vector_dim)
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

    def generate(self, n_samples: int, max_len: int,
                 model=None, device=None, temperature: float = 1.0,
                 distribution: str = 'gaussian', min_len: int = 32) -> torch.Tensor:
        """Generate sequences using random bio-vectors."""
        if model is None:
            raise ValueError("Model is required for random vector control")

        random_vectors = self.generate_random_bio_vectors(n_samples, distribution)
        bio_tensor = torch.tensor(random_vectors, dtype=torch.float32).to(device)

        with torch.no_grad():
            generated = model.generate(
                bio_tensor,
                max_len=max_len,
                temperature=temperature,
                use_gumbel=False,
                min_len=min_len
            )

        return generated


def create_baselines(config: Dict, vocab: Optional[Dict[str, int]] = None):
    """Create all baseline generators from config and vocabulary."""
    vocab_size = config.get('vocab_size', 512)

    if vocab is None:
        # Fallback assumes BOS/EOS/PAD in config and all other ids are content ids.
        bos_id = config.get('bos_token_id', 0)
        eos_id = config.get('eos_token_id', 1)
        pad_id = config.get('pad_token_id', 2)
        token_groups = {
            "bos_id": bos_id,
            "eos_id": eos_id,
            "pad_id": pad_id,
            "shift_ids": [],
            "note_on_by_pitch": {},
            "note_off_by_pitch": {},
            "velocity_ids": [],
            "content_ids": [i for i in range(vocab_size) if i not in {bos_id, eos_id, pad_id}],
        }
    else:
        token_groups = _extract_token_groups(vocab)

    baselines = {
        'random': RandomBaseline(token_groups),
        'markov': MarkovBaseline(token_groups, order=2),
        'unconditional': UnconditionalTransformer(
            vocab_size=vocab_size,
            d_model=config.get('d_model', 256),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 6),
            max_seq_len=config.get('max_seq_len', 512),
            dropout=config.get('dropout', 0.1),
            bos_token_id=token_groups['bos_id'],
            eos_token_id=token_groups['eos_id'],
            pad_token_id=token_groups['pad_id'],
        ),
        'rule_based': RuleBasedGenerator(token_groups),
        'random_vector_control': RandomVectorControl(
            vocab_size=vocab_size,
            bio_vector_dim=config.get('bio_vector_dim', 128)
        )
    }

    return baselines
