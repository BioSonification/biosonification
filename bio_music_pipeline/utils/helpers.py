"""
Utility functions for the bio-music pipeline.

Includes MIDI rendering, gradient checking, and data leak prevention utilities.
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import mido


def tokens_to_midi(tokens: List[int],
                   preprocessor_vocab: Dict[str, int],
                   output_path: str,
                   tempo: int = 120,
                   ticks_per_beat: int = 480) -> bool:
    """
    Convert token sequence to MIDI file.

    Args:
        tokens: List of token IDs
        preprocessor_vocab: Token vocabulary mapping
        output_path: Path to save MIDI file
        tempo: Tempo in BPM
        ticks_per_beat: MIDI time resolution

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create reverse vocabulary
        idx_to_token = {v: k for k, v in preprocessor_vocab.items()}

        # Create MIDI file
        midi = mido.MidiFile()
        midi.ticks_per_beat = ticks_per_beat
        track = mido.MidiTrack()
        midi.tracks.append(track)

        # Set tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))

        current_time = 0
        pending_notes = {}  # pitch -> velocity (waiting for note_off)

        for token_id in tokens:
            if token_id not in idx_to_token:
                continue

            token = idx_to_token[token_id]

            if token.startswith('SHIFT_'):
                shift = int(token.split('_')[1])
                current_time += shift * 10  # Scale back from quantization

            elif token.startswith('VEL_'):
                velocity = int(token.split('_')[1])
                # Apply velocity to all pending notes
                for pitch in list(pending_notes.keys()):
                    track.append(mido.Message('note_on', note=pitch,
                                             velocity=velocity, time=0))
                    pending_notes[pitch] = True  # Mark as played

            elif token.startswith('NOTE_ON_'):
                pitch = int(token.split('_')[2])
                # Store the note, wait for VEL and NOTE_OFF
                pending_notes[pitch] = False  # False = not yet sent

            elif token.startswith('NOTE_OFF_'):
                pitch = int(token.split('_')[2])
                # Send note_off with elapsed time
                track.append(mido.Message('note_off', note=pitch, velocity=64, time=current_time))
                current_time = 0
                if pitch in pending_notes:
                    del pending_notes[pitch]

        # Flush remaining notes
        for pitch in list(pending_notes.keys()):
            track.append(mido.Message('note_off', note=pitch, velocity=64, time=current_time))
            current_time = 0

        # End of track
        track.append(mido.MetaMessage('end_of_track', time=0))

        # Save
        midi.save(output_path)
        return True

    except Exception as e:
        print(f"Error converting tokens to MIDI: {e}")
        import traceback
        traceback.print_exc()
        return False


def batch_tokens_to_midi(sequences: torch.Tensor,
                         vocab: Dict[str, int],
                         output_dir: str,
                         filenames: Optional[List[str]] = None,
                         **midi_kwargs) -> List[str]:
    """
    Convert batch of token sequences to MIDI files.
    
    Args:
        sequences: Tensor of shape (batch_size, seq_len)
        vocab: Token vocabulary
        output_dir: Directory to save MIDI files
        filenames: Optional list of filenames
        **midi_kwargs: Additional arguments for tokens_to_midi
        
    Returns:
        List of created file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    created_files = []
    sequences_np = sequences.cpu().numpy()
    
    for i, seq in enumerate(sequences_np):
        if filenames is not None and i < len(filenames):
            filename = filenames[i]
        else:
            filename = f"generated_{i:04d}.mid"
        
        output_path = output_dir / filename
        tokens_list = seq.tolist()
        
        if tokens_to_midi(tokens_list, vocab, str(output_path), **midi_kwargs):
            created_files.append(str(output_path))
    
    return created_files


def check_gradient_flow(model: torch.nn.Module,
                       sample_input: Dict[str, torch.Tensor],
                       loss_fn=None) -> Dict:
    """
    Verify that gradients flow correctly through the model.
    
    Args:
        model: PyTorch model
        sample_input: Dictionary of input tensors
        loss_fn: Optional custom loss function
        
    Returns:
        Dictionary with gradient flow analysis
    """
    model.train()
    model.zero_grad()
    
    results = {
        'parameters_with_grad': 0,
        'parameters_without_grad': 0,
        'gradient_norms': {},
        'flow_verified': True
    }
    
    try:
        # Forward pass
        if hasattr(model, 'compute_loss'):
            loss_dict = model.compute_loss(**sample_input)
            loss = loss_dict['total_loss']
        else:
            output = model(**sample_input)
            if loss_fn is not None:
                loss = loss_fn(output)
            else:
                # Dummy loss
                if isinstance(output, dict):
                    loss = sum(v.sum() for v in output.values() if isinstance(v, torch.Tensor))
                else:
                    loss = output.sum()
        
        # Backward pass
        loss.backward()
        
        # Check gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    results['parameters_with_grad'] += 1
                    grad_norm = param.grad.norm().item()
                    results['gradient_norms'][name] = float(grad_norm)
                    
                    # Check for vanishing/exploding gradients
                    if grad_norm == 0:
                        results['flow_verified'] = False
                    elif grad_norm > 1000:
                        results['warning'] = f"Large gradient in {name}: {grad_norm}"
                else:
                    results['parameters_without_grad'] += 1
                    results['flow_verified'] = False
            else:
                results['parameters_without_grad'] += 1
        
    except Exception as e:
        results['error'] = str(e)
        results['flow_verified'] = False
    
    return results


def verify_no_data_leak(train_ids: set, test_ids: set) -> bool:
    """
    Verify no data leakage between train and test sets.
    
    Args:
        train_ids: Set of training sample identifiers
        test_ids: Set of test sample identifiers
        
    Returns:
        True if no leak detected, False otherwise
    """
    overlap = train_ids & test_ids
    if len(overlap) > 0:
        print(f"WARNING: Data leak detected! {len(overlap)} samples in both train and test")
        return False
    return True


def create_sample_bio_sequences(n_samples: int = 10,
                                length_range: Tuple[int, int] = (500, 2000),
                                seed: int = 42) -> List[str]:
    """
    Create synthetic biological sequences for testing.
    
    Args:
        n_samples: Number of sequences to generate
        length_range: (min_length, max_length)
        seed: Random seed
        
    Returns:
        List of DNA sequence strings
    """
    np.random.seed(seed)
    nucleotides = ['A', 'C', 'G', 'T']
    
    sequences = []
    for _ in range(n_samples):
        length = np.random.randint(length_range[0], length_range[1] + 1)
        # Generate with varying GC content
        gc_bias = np.random.uniform(0.3, 0.7)
        probs = [
            (1 - gc_bias) / 2,  # A
            gc_bias / 2,         # C
            gc_bias / 2,         # G
            (1 - gc_bias) / 2   # T
        ]
        seq = ''.join(np.random.choice(nucleotides, size=length, p=probs))
        sequences.append(seq)
    
    return sequences


class GradientCheckpoint:
    """Context manager for checking gradient flow during training."""
    
    def __init__(self, model: torch.nn.Module, check_every: int = 100):
        self.model = model
        self.check_every = check_every
        self.step_count = 0
        self.history = []
    
    def check(self, step: int) -> Dict:
        """Perform gradient check at current step."""
        result = {
            'step': step,
            'timestamp': self.step_count
        }
        
        total_norm = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                norm = param.grad.norm().item()
                total_norm += norm ** 2
                if name not in result:
                    result[name] = []
                result[name].append(norm)
        
        result['total_norm'] = float(np.sqrt(total_norm))
        self.history.append(result)
        self.step_count += 1
        
        return result
    
    def should_check(self, step: int) -> bool:
        return step % self.check_every == 0
    
    def get_summary(self) -> Dict:
        """Get summary of gradient history."""
        if not self.history:
            return {}
        
        norms = [h['total_norm'] for h in self.history]
        return {
            'mean_total_norm': float(np.mean(norms)),
            'std_total_norm': float(np.std(norms)),
            'min_norm': float(np.min(norms)),
            'max_norm': float(np.max(norms)),
            'n_checks': len(self.history)
        }
