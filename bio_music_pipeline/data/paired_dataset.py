"""
Music Dataset with Paired Bio-Vectors.

This module provides a dataset class that properly pairs MIDI files
with bio-vectors, ensuring meaningful conditioning during training.
"""

import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .dataset import MusicDataset


class PairedMusicDataset(MusicDataset):
    """
    Music dataset with proper MIDI-bio-vector pairing.

    Instead of random bio-vector assignment, this dataset uses
    pre-computed pairs where each MIDI file has a corresponding
    bio-vector assigned by complexity matching.
    """

    def __init__(self,
                 paired_data_dir: str,
                 midi_base_dir: str = None,
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 test_split: float = 0.15,
                 seed: int = 42,
                 **preprocessor_kwargs):
        """
        Initialize paired music dataset.

        Args:
            paired_data_dir: Directory containing paired_data.json and .npy files
            midi_base_dir: Base directory for MIDI files (if paths are relative)
            train_split: Fraction for training
            val_split: Fraction for validation
            test_split: Fraction for testing
            seed: Random seed for reproducibility
            **preprocessor_kwargs: Arguments for MIDIPreprocessor
        """
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Splits must sum to 1.0"

        self.paired_data_dir = Path(paired_data_dir)
        self.midi_base_dir = Path(midi_base_dir) if midi_base_dir else None

        # Load paired data
        self._load_paired_data()

        # Initialize preprocessor with same settings as parent
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        self.preprocessor = self._create_preprocessor(**preprocessor_kwargs)

        self.train_data = []
        self.val_data = []
        self.test_data = []

        self._split_data()

    def _load_paired_data(self):
        """Load paired data from files."""
        # Load metadata
        with open(self.paired_data_dir / 'paired_data.json', 'r') as f:
            self.paired_metadata = json.load(f)

        # Load bio-vectors
        self.bio_vectors = np.load(self.paired_data_dir / 'paired_bio_vectors.npy')

        # Load conditioning vectors
        self.conditioning_vectors = np.load(self.paired_data_dir / 'paired_conditioning_vectors.npy')

        # Load statistics
        stats_path = self.paired_data_dir / 'paired_stats.json'
        if stats_path.exists():
            with open(stats_path, 'r') as f:
                self.paired_stats = json.load(f)
        else:
            self.paired_stats = {}

        print(f"Loaded paired dataset: {len(self.paired_metadata)} pairs")
        print(f"  Bio-vectors shape: {self.bio_vectors.shape}")
        print(f"  Conditioning vectors shape: {self.conditioning_vectors.shape}")

    def _create_preprocessor(self, **kwargs):
        """Create preprocessor with settings compatible with pipeline."""
        from .dataset import MIDIPreprocessor

        # Get max_seq_len from kwargs or default
        max_seq_len = kwargs.pop('max_seq_len', 512)
        return MIDIPreprocessor(max_seq_len=max_seq_len, **kwargs)

    def _process_midi_file(self, midi_path: str):
        """Process a single MIDI file into token IDs."""
        result = self.preprocessor.process_midi_file(midi_path)
        if result is not None:
            token_ids, duration = result
            return token_ids
        return None

    def _split_data(self):
        """Split paired data into train/val/test."""
        np.random.seed(self.seed)

        # Process all MIDI files and collect valid ones
        valid_pairs = []
        for i, meta in enumerate(self.paired_metadata):
            midi_path = meta['midi_path']

            # Resolve path if relative
            if not Path(midi_path).exists() and self.midi_base_dir:
                midi_path = str(Path(self.midi_base_dir) / Path(midi_path).name)

            if not Path(midi_path).exists():
                print(f"Warning: MIDI file not found: {midi_path}")
                continue

            # Process MIDI
            token_ids = self._process_midi_file(midi_path)
            if token_ids is None:
                continue

            valid_pairs.append({
                'index': i,
                'midi_path': midi_path,
                'token_ids': token_ids,
                'bio_vector': self.bio_vectors[i],
                'conditioning_vector': self.conditioning_vectors[i],
                'metadata': meta
            })

        if len(valid_pairs) == 0:
            print("Warning: No valid paired data found")
            return

        # Shuffle with fixed seed
        indices = np.random.permutation(len(valid_pairs))

        # Calculate split points
        n_total = len(valid_pairs)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)

        # Strict non-overlapping splits
        train_indices = set(indices[:n_train])
        val_indices = set(indices[n_train:n_train + n_val])
        test_indices = set(indices[n_train + n_val:])

        # Assign splits
        self.train_data = [valid_pairs[i] for i in range(len(valid_pairs)) if i in train_indices]
        self.val_data = [valid_pairs[i] for i in range(len(valid_pairs)) if i in val_indices]
        self.test_data = [valid_pairs[i] for i in range(len(valid_pairs)) if i in test_indices]

        print(f"Paired dataset split: {len(self.train_data)} train, "
              f"{len(self.val_data)} val, {len(self.test_data)} test")

    def get_train_loader(self, batch_size: int = 32, shuffle: bool = True):
        """Get training data loader with proper bio-vectors."""
        return self._create_paired_loader(self.train_data, batch_size, shuffle)

    def get_val_loader(self, batch_size: int = 32, shuffle: bool = False):
        """Get validation data loader with proper bio-vectors."""
        return self._create_paired_loader(self.val_data, batch_size, shuffle)

    def get_test_loader(self, batch_size: int = 32, shuffle: bool = False):
        """Get test data loader with proper bio-vectors."""
        return self._create_paired_loader(self.test_data, batch_size, shuffle)

    def _create_paired_loader(self, data: List, batch_size: int, shuffle: bool):
        """Create batch generator that yields (midi_path, tokens, bio_vector, conditioning_vector)."""
        if len(data) == 0:
            return

        indices = list(range(len(data)))
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(data), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch = [data[i] for i in batch_indices]
            yield batch

    def get_test_data(self) -> List:
        """Get test data for evaluation."""
        return self.test_data.copy()

    def get_bio_vector_for_sample(self, sample_idx: int, split: str = 'train') -> np.ndarray:
        """Get the bio-vector for a specific sample."""
        if split == 'train':
            data = self.train_data
        elif split == 'val':
            data = self.val_data
        else:
            data = self.test_data

        if 0 <= sample_idx < len(data):
            return data[sample_idx]['bio_vector']
        return None

    def save_splits(self, output_dir: str):
        """Save split information."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        splits = {
            'train': [d['midi_path'] for d in self.train_data],
            'val': [d['midi_path'] for d in self.val_data],
            'test': [d['midi_path'] for d in self.test_data]
        }

        for split_name, files in splits.items():
            filepath = output_dir / f"{split_name}_files.txt"
            with open(filepath, 'w') as f:
                for filepath in files:
                    f.write(f"{filepath}\n")

        # Save metadata
        metadata = {
            'n_train': len(self.train_data),
            'n_val': len(self.val_data),
            'n_test': len(self.test_data),
            'vocab_size': self.preprocessor.vocab_size,
            'max_seq_len': self.preprocessor.max_seq_len,
            'bio_vector_dim': self.bio_vectors.shape[1],
            'conditioning_dim': self.conditioning_vectors.shape[1]
        }

        with open(output_dir / 'paired_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
