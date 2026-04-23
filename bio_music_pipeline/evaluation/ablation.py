"""
Ablation Study Module for Bio-Music Pipeline.

Provides tools to study the contribution of individual bio-vector
components to music generation quality.

Ablation types:
- 'nuc_freqs': Remove nucleotide frequencies (4 dims)
- 'entropy': Remove Shannon entropy (1 dim)
- 'skews': Remove GC/AT skew (2 dims)
- 'kmer_1': Remove unigram k-mers (4 dims)
- 'kmer_2': Remove bigram k-mers (16 dims)
- 'kmer_3': Remove trigram k-mers (64 dims)
- 'windowed': Remove windowed statistics (4 dims)
- 'summary': Remove summary statistics (4 dims)
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from copy import deepcopy


# Bio-vector component indices (standard ordering from BioVectorExtractor)
BIO_VECTOR_COMPONENTS = {
    'nuc_freqs': (0, 4),       # indices 0-3: nucleotide frequencies
    'entropy': (4, 5),         # index 4: Shannon entropy
    'gc_skew': (5, 6),         # index 5: GC skew
    'at_skew': (6, 7),         # index 6: AT skew
    'kmer_1': (7, 11),         # indices 7-10: k=1 distribution
    'kmer_2': (11, 27),        # indices 11-26: k=2 distribution
    'kmer_3': (27, 91),        # indices 27-90: k=3 distribution
    'gc_window_mean': (91, 92),
    'gc_window_std': (92, 93),
    'entropy_window_mean': (93, 94),
    'entropy_window_std': (94, 95),
    # Summary stats (last 4 elements when padded to 128)
    'summary_mean': (124, 125),
    'summary_std': (125, 126),
    'summary_min': (126, 127),
    'summary_max': (127, 128),
}

# Ablation groups for convenience
ABLATION_GROUPS = {
    'no_nuc_freqs': ['nuc_freqs'],
    'no_entropy': ['entropy'],
    'no_skews': ['gc_skew', 'at_skew'],
    'no_kmer_1': ['kmer_1'],
    'no_kmer_2': ['kmer_2'],
    'no_kmer_3': ['kmer_3'],
    'no_kmers': ['kmer_1', 'kmer_2', 'kmer_3'],
    'no_windowed': ['gc_window_mean', 'gc_window_std', 'entropy_window_mean', 'entropy_window_std'],
    'no_summary': ['summary_mean', 'summary_std', 'summary_min', 'summary_max'],
}


def ablate_bio_vector(bio_vector: np.ndarray,
                       components: List[str],
                       replacement: str = 'zero') -> np.ndarray:
    """
    Ablate specific components from a bio-vector.

    Args:
        bio_vector: Original bio-vector (128 dim)
        components: List of component names to ablate
        replacement: How to replace ablated values ('zero', 'mean', 'noise')

    Returns:
        Ablated bio-vector
    """
    ablated = bio_vector.copy()

    for component in components:
        if component in BIO_VECTOR_COMPONENTS:
            start, end = BIO_VECTOR_COMPONENTS[component]
            start = min(start, len(ablated))
            end = min(end, len(ablated))

            if replacement == 'zero':
                ablated[start:end] = 0.0
            elif replacement == 'mean':
                ablated[start:end] = np.mean(bio_vector)
            elif replacement == 'noise':
                ablated[start:end] = np.random.randn(end - start) * 0.1
            else:
                raise ValueError(f"Unknown replacement: {replacement}")

    return ablated


def create_ablated_dataset(bio_vectors: np.ndarray,
                           ablation_type: str,
                           replacement: str = 'zero') -> np.ndarray:
    """
    Create ablated version of entire bio-vector dataset.

    Args:
        bio_vectors: Array of shape (n_samples, 128)
        ablation_type: Name from ABLATION_GROUPS or custom list
        replacement: Replacement strategy

    Returns:
        Ablated bio-vectors
    """
    if ablation_type in ABLATION_GROUPS:
        components = ABLATION_GROUPS[ablation_type]
    elif isinstance(ablation_type, list):
        components = ablation_type
    else:
        components = [ablation_type]

    ablated = np.zeros_like(bio_vectors)
    for i, vec in enumerate(bio_vectors):
        ablated[i] = ablate_bio_vector(vec, components, replacement)

    return ablated


def run_ablation_study(model,
                       bio_vectors: np.ndarray,
                       device: torch.device,
                       max_len: int = 512,
                       n_samples: int = 25,
                       temperature: float = 1.0,
                       ablation_types: List[str] = None) -> Dict:
    """
    Run ablation study on trained model.

    For each ablation type:
    1. Create ablated bio-vectors
    2. Generate sequences
    3. Compare quality to full bio-vector generation

    Args:
        model: Trained BioConditionedTransformerDecoder
        bio_vectors: Original bio-vectors
        device: Device to run on
        max_len: Maximum generation length
        n_samples: Number of samples per ablation
        temperature: Sampling temperature
        ablation_types: List of ablation types to test

    Returns:
        Dictionary with ablation results
    """
    model.eval()

    if ablation_types is None:
        ablation_types = list(ABLATION_GROUPS.keys())

    results = {}

    # Generate baseline with full bio-vectors
    print("Generating baseline (full bio-vectors)...")
    bio_subset = bio_vectors[:n_samples]
    bio_tensor = torch.tensor(bio_subset, dtype=torch.float32).to(device)

    with torch.no_grad():
        baseline_seqs = model.generate(
            bio_tensor, max_len=max_len,
            temperature=temperature, use_gumbel=False
        )
    baseline_seqs = baseline_seqs.cpu().numpy()

    # Compute baseline statistics
    baseline_stats = compute_sequence_statistics(baseline_seqs)

    # Run each ablation
    for ablation_type in ablation_types:
        print(f"\nRunning ablation: {ablation_type}")

        # Create ablated vectors
        ablated_vectors = create_ablated_dataset(bio_subset, ablation_type)
        ablated_tensor = torch.tensor(ablated_vectors, dtype=torch.float32).to(device)

        # Generate
        with torch.no_grad():
            ablated_seqs = model.generate(
                ablated_tensor, max_len=max_len,
                temperature=temperature, use_gumbel=False
            )
        ablated_seqs = ablated_seqs.cpu().numpy()

        # Compute statistics
        ablated_stats = compute_sequence_statistics(ablated_seqs)

        # Compare to baseline
        comparison = compare_statistics(baseline_stats, ablated_stats)

        results[ablation_type] = {
            'ablated_vectors_shape': ablated_vectors.shape,
            'generated_sequences_shape': ablated_seqs.shape,
            'baseline_statistics': baseline_stats,
            'ablated_statistics': ablated_stats,
            'comparison': comparison
        }

    return results


def compute_sequence_statistics(sequences: np.ndarray) -> Dict:
    """Compute statistical features of generated sequences."""
    stats = {
        'mean_token': [],
        'std_token': [],
        'unique_tokens': [],
        'sequence_length': [],
        'eos_position': [],  # Position of first EOS token
        'note_density': [],
    }

    for seq in sequences:
        # Remove padding (tokens == 1 are typically EOS/PAD)
        valid_mask = seq > 1
        valid_tokens = seq[valid_mask]

        if len(valid_tokens) == 0:
            continue

        stats['mean_token'].append(float(np.mean(valid_tokens)))
        stats['std_token'].append(float(np.std(valid_tokens)))
        stats['unique_tokens'].append(float(len(set(valid_tokens))))
        stats['sequence_length'].append(len(valid_tokens))

        # Find EOS position
        eos_positions = np.where(seq == 1)[0]
        if len(eos_positions) > 0:
            stats['eos_position'].append(int(eos_positions[0]))
        else:
            stats['eos_position'].append(len(seq))

        # Note density (approximate: count NOTE_ON tokens)
        # NOTE_ON tokens are typically in range [vocab_offset, vocab_offset+88]
        note_count = np.sum((seq >= 2) & (seq < 103))
        stats['note_density'].append(float(note_count / max(len(seq), 1)))

    # Compute aggregates
    for key in stats:
        if stats[key]:
            stats[key + '_mean'] = float(np.mean(stats[key]))
            stats[key + '_std'] = float(np.std(stats[key]))
        else:
            stats[key + '_mean'] = 0.0
            stats[key + '_std'] = 0.0

    return stats


def compare_statistics(baseline: Dict, ablated: Dict) -> Dict:
    """Compare ablated statistics to baseline."""
    comparison = {}

    for key in ['mean_token_mean', 'std_token_mean', 'unique_tokens_mean',
                'sequence_length_mean', 'note_density_mean']:
        if key in baseline and key in ablated:
            baseline_val = baseline[key]
            ablated_val = ablated[key]
            diff = ablated_val - baseline_val
            rel_diff = diff / max(abs(baseline_val), 1e-10)
            comparison[key] = {
                'baseline': baseline_val,
                'ablated': ablated_val,
                'diff': diff,
                'rel_diff': rel_diff
            }

    return comparison
