"""
Musical Quality Metrics for Bio-Music Pipeline.

Provides specialized metrics for evaluating the musical quality
of generated sequences beyond simple cross-entropy loss:

- Tonal consonance / stability
- Rhythmic regularity
- Melodic contour analysis
- Harmonic complexity
- Self-similarity / repetitivity
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import Counter


# MIDI note to pitch class mapping
PITCH_CLASS_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Major and minor scale intervals (semitones from root)
MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
PENTATONIC_MAJOR = [0, 2, 4, 7, 9]
PENTATONIC_MINOR = [0, 3, 5, 7, 10]
BLUES_SCALE = [0, 3, 5, 6, 7, 10]

# Consonant intervals (in semitones)
CONSONANT_INTERVALS = [0, 3, 4, 5, 7, 8, 9]  # Unison, minor/major 3rd, perfect 4th/5th, minor/major 6th
DISSONANT_INTERVALS = [1, 2, 6, 10, 11]  # minor/major 2nd, tritone, minor/major 7th


class MusicalQualityMetrics:
    """
    Compute musical quality metrics for generated sequences.

    Metrics focus on:
    1. Tonal stability (do notes fit a consistent key?)
    2. Rhythmic regularity (are timing patterns consistent?)
    3. Melodic coherence (do melodic intervals make musical sense?)
    4. Harmonic richness (is there chordal variety?)
    5. Structural coherence (are there repeating patterns?)
    """

    def __init__(self, vocab: Dict[str, int] = None):
        """
        Initialize metrics calculator.

        Args:
            vocab: Token vocabulary for decoding tokens to notes
        """
        self.vocab = vocab
        if vocab:
            self.idx_to_token = {v: k for k, v in vocab.items()}
        else:
            self.idx_to_token = None

    def decode_tokens_to_notes(self, tokens: np.ndarray) -> List[Dict]:
        """
        Decode token sequence to list of note events.

        Returns:
            List of dicts with keys: pitch, velocity, time, type
        """
        if self.idx_to_token is None:
            raise ValueError("Vocabulary not provided")

        notes = []
        current_time = 0
        current_velocity = 64
        active_notes = {}

        for token_id in tokens:
            if token_id not in self.idx_to_token:
                continue

            token = self.idx_to_token[token_id]

            if token.startswith('SHIFT_'):
                shift = int(token.split('_')[1])
                current_time += shift * 10

            elif token.startswith('VEL_'):
                current_velocity = int(token.split('_')[1])

            elif token.startswith('NOTE_ON_'):
                pitch = int(token.split('_')[2])
                active_notes[pitch] = (current_time, current_velocity)

            elif token.startswith('NOTE_OFF_'):
                pitch = int(token.split('_')[2])
                if pitch in active_notes:
                    start_time, velocity = active_notes.pop(pitch)
                    notes.append({
                        'pitch': pitch,
                        'velocity': velocity,
                        'start_time': start_time,
                        'duration': current_time - start_time,
                        'pitch_class': pitch % 12
                    })

        return notes

    def compute_tonal_stability(self, notes: List[Dict]) -> Dict:
        """
        Compute tonal stability metrics.

        Measures how well the notes fit a consistent musical key.

        Returns:
            Dictionary with tonal stability metrics
        """
        if not notes:
            return {
                'tonal_stability': 0.0,
                'best_key': None,
                'key_fit_score': 0.0,
                'scale_conformance': 0.0
            }

        pitch_classes = [n['pitch_class'] for n in notes]
        pc_counts = Counter(pitch_classes)
        total_notes = len(pitch_classes)

        # Try all 12 keys for major and minor scales
        best_key = None
        best_score = -1

        for root in range(12):
            # Major scale
            major_pcs = set((root + interval) % 12 for interval in MAJOR_SCALE)
            major_fit = sum(pc_counts.get(pc, 0) for pc in major_pcs) / total_notes

            # Minor scale
            minor_pcs = set((root + interval) % 12 for interval in MINOR_SCALE)
            minor_fit = sum(pc_counts.get(pc, 0) for pc in minor_pcs) / total_notes

            if major_fit > best_score:
                best_score = major_fit
                best_key = f"{PITCH_CLASS_NAMES[root]}_major"

            if minor_fit > best_score:
                best_score = minor_fit
                best_key = f"{PITCH_CLASS_NAMES[root]}_minor"

        # Compute scale conformance (fraction of notes in best scale)
        if 'major' in best_key:
            root = PITCH_CLASS_NAMES.index(best_key.split('_')[0])
            scale_pcs = set((root + interval) % 12 for interval in MAJOR_SCALE)
        else:
            root = PITCH_CLASS_NAMES.index(best_key.split('_')[0])
            scale_pcs = set((root + interval) % 12 for interval in MINOR_SCALE)

        scale_conformance = sum(1 for pc in pitch_classes if pc in scale_pcs) / total_notes

        return {
            'tonal_stability': float(best_score),
            'best_key': best_key,
            'key_fit_score': float(best_score),
            'scale_conformance': float(scale_conformance)
        }

    def compute_rhythmic_regular(self, notes: List[Dict]) -> Dict:
        """
        Compute rhythmic regularity metrics.

        Measures consistency of timing patterns.

        Returns:
            Dictionary with rhythmic metrics
        """
        if len(notes) < 2:
            return {
                'rhythmic_regularity': 0.0,
                'mean_ioi': 0.0,
                'ioi_cv': 0.0,  # coefficient of variation
                'tempo_consistency': 0.0
            }

        # Compute inter-onset intervals (IOI)
        onsets = [n['start_time'] for n in notes]
        iois = np.diff(onsets)
        iois = iois[iois > 0]  # Remove zero-duration intervals

        if len(iois) < 2:
            return {
                'rhythmic_regularity': 0.0,
                'mean_ioi': float(np.mean(iois)) if len(iois) > 0 else 0.0,
                'ioi_cv': 0.0,
                'tempo_consistency': 0.0
            }

        mean_ioi = np.mean(iois)
        std_ioi = np.std(iois)
        cv = std_ioi / max(mean_ioi, 1)  # Coefficient of variation (lower = more regular)

        # Rhythmic regularity: inverse of CV, normalized to [0, 1]
        regularity = 1.0 / (1.0 + cv)

        # Quantize IOIs to common rhythmic values and measure fit
        common_ratios = [1, 2, 3, 4, 6, 8]  # Simple rhythmic ratios
        min_ioi = np.min(iois)
        if min_ioi > 0:
            quantized_iois = []
            for ioi in iois:
                ratio = ioi / min_ioi
                closest = min(common_ratios, key=lambda x: abs(x - ratio))
                quantized_iois.append(closest * min_ioi)

            quantization_error = np.mean(np.abs(np.array(iois) - np.array(quantized_iois)))
            tempo_consistency = 1.0 / (1.0 + quantization_error / max(min_ioi, 1))
        else:
            tempo_consistency = 0.0

        return {
            'rhythmic_regularity': float(regularity),
            'mean_ioi': float(mean_ioi),
            'ioi_cv': float(cv),
            'tempo_consistency': float(tempo_consistency)
        }

    def compute_melodic_coherence(self, notes: List[Dict]) -> Dict:
        """
        Compute melodic coherence metrics.

        Measures whether melodic intervals follow musical conventions.

        Returns:
            Dictionary with melodic metrics
        """
        if len(notes) < 2:
            return {
                'melodic_coherence': 0.0,
                'mean_interval': 0.0,
                'consonant_interval_ratio': 0.0,
                'stepwise_motion': 0.0
            }

        pitches = [n['pitch'] for n in notes]
        intervals = np.abs(np.diff(pitches))

        # Mean interval size
        mean_interval = np.mean(intervals)

        # Consonant interval ratio
        consonant_count = sum(1 for i in intervals if (i % 12) in CONSONANT_INTERVALS)
        consonant_ratio = consonant_count / len(intervals) if len(intervals) > 0 else 0

        # Stepwise motion (intervals ≤ 3 semitones)
        stepwise_count = sum(1 for i in intervals if i <= 3)
        stepwise_ratio = stepwise_count / len(intervals) if len(intervals) > 0 else 0

        # Overall coherence
        coherence = 0.4 * consonant_ratio + 0.3 * stepwise_ratio + 0.3 * max(0, 1 - mean_interval / 12)

        return {
            'melodic_coherence': float(coherence),
            'mean_interval': float(mean_interval),
            'consonant_interval_ratio': float(consonant_ratio),
            'stepwise_motion': float(stepwise_ratio)
        }

    def compute_harmonic_richness(self, notes: List[Dict]) -> Dict:
        """
        Compute harmonic richness metrics.

        Measures pitch variety and chordal content.

        Returns:
            Dictionary with harmonic metrics
        """
        if not notes:
            return {
                'harmonic_richness': 0.0,
                'pitch_range': 0,
                'unique_pitches': 0,
                'pitch_entropy': 0.0
            }

        pitches = [n['pitch'] for n in notes]
        unique_pitches = len(set(pitches))
        pitch_range = max(pitches) - min(pitches) if len(pitches) > 1 else 0

        # Pitch class entropy
        pitch_classes = [n['pitch_class'] for n in notes]
        pc_counts = Counter(pitch_classes)
        total = len(pitch_classes)
        probs = [count / total for count in pc_counts.values()]
        pitch_entropy = -sum(p * np.log2(p) for p in probs if p > 0)

        # Normalize entropy to [0, 1]
        max_entropy = np.log2(min(12, unique_pitches)) if unique_pitches > 1 else 1
        normalized_entropy = pitch_entropy / max(max_entropy, 1)

        # Richness combines variety and entropy
        richness = 0.4 * min(unique_pitches / 20, 1) + 0.3 * min(pitch_range / 48, 1) + 0.3 * normalized_entropy

        return {
            'harmonic_richness': float(richness),
            'pitch_range': int(pitch_range),
            'unique_pitches': int(unique_pitches),
            'pitch_entropy': float(pitch_entropy)
        }

    def compute_self_similarity(self, notes: List[Dict], window: int = 20) -> Dict:
        """
        Compute self-similarity / repetitivity.

        Measures whether the music contains repeating patterns.

        Returns:
            Dictionary with self-similarity metrics
        """
        if len(notes) < window * 2:
            return {
                'self_similarity': 0.0,
                'max_similarity': 0.0,
                'repetition_rate': 0.0
            }

        # Use pitch classes for comparison
        pitch_seq = np.array([n['pitch_class'] for n in notes])

        # Compute autocorrelation at different lags
        similarities = []
        n_windows = len(pitch_seq) - window

        for lag in range(window, n_windows, window // 2):
            seq1 = pitch_seq[:min(window, n_windows - lag)]
            seq2 = pitch_seq[lag:lag + len(seq1)]

            if len(seq1) > 0:
                similarity = np.mean(seq1 == seq2)
                similarities.append(similarity)

        if not similarities:
            return {
                'self_similarity': 0.0,
                'max_similarity': 0.0,
                'repetition_rate': 0.0
            }

        max_similarity = max(similarities)
        mean_similarity = np.mean(similarities)

        # Repetition rate: fraction of windows with high similarity
        repetition_rate = np.mean([s > 0.5 for s in similarities])

        return {
            'self_similarity': float(mean_similarity),
            'max_similarity': float(max_similarity),
            'repetition_rate': float(repetition_rate)
        }

    def compute_all_metrics(self, sequences: np.ndarray) -> Dict[str, List]:
        """
        Compute all musical quality metrics for a batch of sequences.

        Args:
            sequences: Array of shape (n_sequences, seq_len)

        Returns:
            Dictionary with metric names and values for each sequence
        """
        all_metrics = {
            'tonal_stability': [],
            'key_fit_score': [],
            'scale_conformance': [],
            'rhythmic_regularity': [],
            'tempo_consistency': [],
            'melodic_coherence': [],
            'consonant_interval_ratio': [],
            'stepwise_motion': [],
            'harmonic_richness': [],
            'pitch_range': [],
            'unique_pitches': [],
            'self_similarity': [],
            'max_similarity': [],
            'repetition_rate': [],
        }

        for seq in sequences:
            if self.idx_to_token:
                notes = self.decode_tokens_to_notes(seq)
            else:
                # Simple fallback: treat non-zero tokens as "notes"
                notes = [{'pitch': int(t), 'pitch_class': int(t) % 12,
                         'start_time': i, 'duration': 1, 'velocity': 64}
                        for i, t in enumerate(seq) if t > 1]

            if not notes:
                # Fill with zeros
                for key in all_metrics:
                    all_metrics[key].append(0.0)
                continue

            # Tonal stability
            tonal = self.compute_tonal_stability(notes)
            all_metrics['tonal_stability'].append(tonal['tonal_stability'])
            all_metrics['key_fit_score'].append(tonal['key_fit_score'])
            all_metrics['scale_conformance'].append(tonal['scale_conformance'])

            # Rhythmic regularity
            rhythmic = self.compute_rhythmic_regular(notes)
            all_metrics['rhythmic_regularity'].append(rhythmic['rhythmic_regularity'])
            all_metrics['tempo_consistency'].append(rhythmic['tempo_consistency'])

            # Melodic coherence
            melodic = self.compute_melodic_coherence(notes)
            all_metrics['melodic_coherence'].append(melodic['melodic_coherence'])
            all_metrics['consonant_interval_ratio'].append(melodic['consonant_interval_ratio'])
            all_metrics['stepwise_motion'].append(melodic['stepwise_motion'])

            # Harmonic richness
            harmonic = self.compute_harmonic_richness(notes)
            all_metrics['harmonic_richness'].append(harmonic['harmonic_richness'])
            all_metrics['pitch_range'].append(harmonic['pitch_range'])
            all_metrics['unique_pitches'].append(harmonic['unique_pitches'])

            # Self-similarity
            similarity = self.compute_self_similarity(notes)
            all_metrics['self_similarity'].append(similarity['self_similarity'])
            all_metrics['max_similarity'].append(similarity['max_similarity'])
            all_metrics['repetition_rate'].append(similarity['repetition_rate'])

        return all_metrics

    def compute_mean_metrics(self, sequences: np.ndarray) -> Dict[str, float]:
        """Compute mean values for all metrics."""
        all_metrics = self.compute_all_metrics(sequences)
        mean_metrics = {}
        for key, values in all_metrics.items():
            if values:
                mean_metrics[key] = float(np.mean(values))
            else:
                mean_metrics[key] = 0.0
        return mean_metrics


def compare_musical_quality(generated_samples: Dict[str, np.ndarray],
                             vocab: Dict[str, int]) -> Dict:
    """
    Compare musical quality across different generation conditions.

    Args:
        generated_samples: Dict mapping condition to sequences
        vocab: Token vocabulary

    Returns:
        Dictionary with quality comparison
    """
    metrics_calc = MusicalQualityMetrics(vocab)
    results = {}

    for condition, sequences in generated_samples.items():
        print(f"Computing metrics for: {condition}")
        results[condition] = metrics_calc.compute_mean_metrics(sequences)

    # Compute improvements over random baseline
    if 'random' in results:
        baseline = results['random']
        for condition in results:
            if condition != 'random':
                results[condition]['_improvement_over_random'] = {}
                for key in results[condition]:
                    if key in baseline and baseline[key] != 0:
                        improvement = (results[condition][key] - baseline[key]) / baseline[key]
                        results[condition]['_improvement_over_random'][key] = improvement

    return results
