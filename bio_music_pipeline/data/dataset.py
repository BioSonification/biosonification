"""
Stage 3: Music dataset preparation with strict train/val/test splits.

This module handles loading, preprocessing, and splitting of MIDI music data.
Ensures non-overlapping splits and proper handling of edge cases.
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import mido


class MIDIPreprocessor:
    """
    Preprocesses MIDI files into token sequences for model training.
    """
    
    def __init__(self, 
                 max_seq_len: int = 512,
                 min_duration: float = 30.0,
                 max_duration: float = 300.0,
                 pitch_range: Tuple[int, int] = (21, 108),
                 time_resolution: int = 480):
        """
        Initialize MIDI preprocessor.
        
        Args:
            max_seq_len: Maximum sequence length for tokens
            min_duration: Minimum MIDI duration in seconds
            max_duration: Maximum MIDI duration in seconds
            pitch_range: Valid MIDI pitch range (min, max)
            time_resolution: MIDI ticks per beat
        """
        self.max_seq_len = max_seq_len
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.pitch_range = pitch_range
        self.time_resolution = time_resolution
        
        # Token vocabulary
        self.vocab = self._build_vocab()
        self.token_to_idx = {t: i for i, t in enumerate(self.vocab)}
        self.idx_to_token = {i: t for i, t in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        self.bos_token_id = self.token_to_idx["BOS"]
        self.eos_token_id = self.token_to_idx["EOS"]
        self.pad_token_id = self.token_to_idx["PAD"]
    
    def _build_vocab(self) -> List[str]:
        """Build token vocabulary."""
        tokens = []
        
        # Note events: NOTE_ON_pitch, NOTE_OFF_pitch
        for pitch in range(self.pitch_range[0], self.pitch_range[1] + 1):
            tokens.append(f"NOTE_ON_{pitch}")
            tokens.append(f"NOTE_OFF_{pitch}")
        
        # Time shift events: SHIFT_0 to SHIFT_max_shift
        max_shift = 100  # Maximum time shift in ticks
        for i in range(max_shift + 1):
            tokens.append(f"SHIFT_{i}")
        
        # Velocity events: VEL_0 to VEL_127
        for i in range(128):
            tokens.append(f"VEL_{i}")
        
        # Special tokens
        tokens.append("BOS")  # Beginning of sequence
        tokens.append("EOS")  # End of sequence
        tokens.append("PAD")  # Padding
        
        return tokens
    
    def get_bos_token(self) -> str:
        return "BOS"
    
    def get_eos_token(self) -> str:
        return "EOS"
    
    def get_pad_token(self) -> str:
        return "PAD"

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def get_eos_token_id(self) -> int:
        return self.eos_token_id

    def get_pad_token_id(self) -> int:
        return self.pad_token_id
    
    def load_midi_file(self, filepath: str) -> Optional[mido.MidiFile]:
        """Load a MIDI file with error handling."""
        try:
            midi = mido.MidiFile(filepath)
            return midi
        except Exception as e:
            print(f"Warning: Could not load MIDI file {filepath}: {e}")
            return None
    
    def midi_to_events(self, midi: mido.MidiFile) -> List[Dict]:
        """
        Convert MIDI file to list of musical events.
        
        Args:
            midi: Loaded MIDI file
            
        Returns:
            List of event dictionaries
        """
        events = []
        current_time = 0
        
        # Merge all tracks
        merged_track = mido.merge_tracks(midi.tracks)
        
        for msg in merged_track:
            current_time += msg.time
            
            if msg.type == 'note_on' and msg.velocity > 0:
                if self.pitch_range[0] <= msg.note <= self.pitch_range[1]:
                    events.append({
                        'type': 'note_on',
                        'pitch': msg.note,
                        'velocity': msg.velocity,
                        'time': current_time
                    })
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if self.pitch_range[0] <= msg.note <= self.pitch_range[1]:
                    events.append({
                        'type': 'note_off',
                        'pitch': msg.note,
                        'time': current_time
                    })
        
        return events
    
    def events_to_tokens(self, events: List[Dict]) -> List[str]:
        """
        Convert event list to token sequence.
        
        Args:
            events: List of musical events
            
        Returns:
            List of token strings
        """
        tokens = [self.get_bos_token()]
        
        if len(events) == 0:
            tokens.append(self.get_eos_token())
            return tokens
        
        prev_time = 0
        
        for event in events:
            # Time shift
            time_diff = event['time'] - prev_time
            # Quantize time to nearest tick unit (simplify to units of 10 ticks)
            shift_units = min(time_diff // 10, 100)
            shift_units = max(0, shift_units)
            tokens.append(f"SHIFT_{shift_units}")
            
            if event['type'] == 'note_on':
                tokens.append(f"NOTE_ON_{event['pitch']}")
                tokens.append(f"VEL_{event['velocity']}")
            elif event['type'] == 'note_off':
                tokens.append(f"NOTE_OFF_{event['pitch']}")
            
            prev_time = event['time']
        
        tokens.append(self.get_eos_token())
        return tokens
    
    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert token strings to vocabulary indices."""
        ids = []
        for token in tokens:
            if token in self.token_to_idx:
                ids.append(self.token_to_idx[token])
            else:
                # Handle unknown tokens (skip or use PAD)
                pass
        return ids
    
    def process_midi_file(self, filepath: str) -> Optional[Tuple[List[int], float]]:
        """
        Process single MIDI file into token IDs.
        
        Args:
            filepath: Path to MIDI file
            
        Returns:
            Tuple of (token_ids, duration) or None if invalid
        """
        midi = self.load_midi_file(filepath)
        if midi is None:
            return None
        
        # Calculate duration in seconds using tempo map.
        merged = mido.merge_tracks(midi.tracks)
        if midi.ticks_per_beat > 0:
            current_tempo = mido.bpm2tempo(120)  # default tempo if not specified
            duration_sec = 0.0
            for msg in merged:
                duration_sec += mido.tick2second(msg.time, midi.ticks_per_beat, current_tempo)
                if msg.type == 'set_tempo':
                    current_tempo = msg.tempo
        else:
            duration_sec = 0.0
        
        # Check duration constraints
        if duration_sec < self.min_duration or duration_sec > self.max_duration:
            return None
        
        # Convert to events then tokens
        events = self.midi_to_events(midi)
        if len(events) == 0:
            return None
        
        tokens = self.events_to_tokens(events)
        
        # Truncate if too long
        if len(tokens) > self.max_seq_len:
            tokens = tokens[:self.max_seq_len - 1] + [self.get_eos_token()]
        
        token_ids = self.tokens_to_ids(tokens)
        
        if len(token_ids) < 2:  # At least BOS and EOS
            return None
        
        return token_ids, duration_sec
    
    def process_directory(self, directory: str) -> List[Tuple[str, List[int], float]]:
        """
        Process all MIDI files in directory.
        
        Args:
            directory: Path to directory containing MIDI files
            
        Returns:
            List of (filepath, token_ids, duration) tuples
        """
        results = []
        directory = Path(directory)
        
        for filepath in directory.rglob("*.mid"):
            result = self.process_midi_file(str(filepath))
            if result is not None:
                token_ids, duration = result
                results.append((str(filepath), token_ids, duration))
        
        for filepath in directory.rglob("*.midi"):
            result = self.process_midi_file(str(filepath))
            if result is not None:
                token_ids, duration = result
                results.append((str(filepath), token_ids, duration))
        
        return results


class MusicDataset:
    """
    Music dataset with train/val/test splits.
    
    Ensures strict non-overlapping splits and provides data loaders.
    """
    
    def __init__(self, 
                 data_dir: str,
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 test_split: float = 0.15,
                 seed: int = 42,
                 **preprocessor_kwargs):
        """
        Initialize music dataset.
        
        Args:
            data_dir: Directory containing MIDI files
            train_split: Fraction for training
            val_split: Fraction for validation
            test_split: Fraction for testing
            seed: Random seed for reproducibility
            **preprocessor_kwargs: Arguments for MIDIPreprocessor
        """
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
            "Splits must sum to 1.0"
        
        self.data_dir = Path(data_dir)
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        
        self.preprocessor = MIDIPreprocessor(**preprocessor_kwargs)
        
        self.train_data = []
        self.val_data = []
        self.test_data = []
        
        self._load_and_split()
    
    def _load_and_split(self):
        """Load and split dataset."""
        np.random.seed(self.seed)
        
        # Process all MIDI files
        all_data = self.preprocessor.process_directory(str(self.data_dir))
        
        if len(all_data) == 0:
            print("Warning: No valid MIDI files found in data directory")
            return
        
        # Shuffle with fixed seed
        indices = np.random.permutation(len(all_data))
        
        # Calculate split points
        n_total = len(all_data)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)
        
        # Strict non-overlapping splits
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Verify no overlap
        train_set = set(train_indices)
        val_set = set(val_indices)
        test_set = set(test_indices)
        
        assert len(train_set & val_set) == 0, "Train/val overlap detected"
        assert len(train_set & test_set) == 0, "Train/test overlap detected"
        assert len(val_set & test_set) == 0, "Val/test overlap detected"
        
        # Assign splits
        self.train_data = [all_data[i] for i in train_indices]
        self.val_data = [all_data[i] for i in val_indices]
        self.test_data = [all_data[i] for i in test_indices]
        
        print(f"Dataset loaded: {n_train} train, {n_val} val, {len(test_indices)} test")
    
    def get_train_loader(self, batch_size: int = 32, shuffle: bool = True):
        """Get training data loader."""
        return self._create_loader(self.train_data, batch_size, shuffle)
    
    def get_val_loader(self, batch_size: int = 32, shuffle: bool = False):
        """Get validation data loader."""
        return self._create_loader(self.val_data, batch_size, shuffle)
    
    def get_test_loader(self, batch_size: int = 32, shuffle: bool = False):
        """Get test data loader."""
        return self._create_loader(self.test_data, batch_size, shuffle)
    
    def _create_loader(self, data: List, batch_size: int, shuffle: bool):
        """Create simple batch generator."""
        if len(data) == 0:
            return
        
        indices = list(range(len(data)))
        if shuffle:
            np.random.shuffle(indices)
        
        for start_idx in range(0, len(data), batch_size):
            batch_indices = indices[start_idx:start_idx + batch_size]
            batch = [data[i] for i in batch_indices]
            yield batch
    
    def get_test_data(self) -> List[Tuple[str, List[int], float]]:
        """Get test data for evaluation (no shuffling)."""
        return self.test_data.copy()
    
    def save_splits(self, output_dir: str):
        """Save split information to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        splits = {
            'train': [d[0] for d in self.train_data],
            'val': [d[0] for d in self.val_data],
            'test': [d[0] for d in self.test_data]
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
            'max_seq_len': self.preprocessor.max_seq_len
        }
        
        with open(output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


def prepare_music_dataset(data_dir: str, 
                          output_dir: str,
                          **kwargs) -> MusicDataset:
    """
    Convenience function to prepare and save music dataset.
    
    Args:
        data_dir: Directory with MIDI files
        output_dir: Directory to save processed data
        **kwargs: Additional arguments for MusicDataset
        
    Returns:
        Prepared MusicDataset instance
    """
    dataset = MusicDataset(data_dir=data_dir, **kwargs)
    dataset.save_splits(output_dir)
    return dataset
