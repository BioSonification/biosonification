"""
Stage 2: Deterministic sonification rules.

This module transforms bio-vectors into explicit musical parameters through
transparent, deterministic mapping rules. No causal claims are made - these
are structured conditioning signals that statistically influence music structure.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MusicalParameters:
    """Container for mapped musical parameters."""
    key: str  # Musical key (e.g., "C_major", "A_minor")
    tempo: float  # BPM
    pitch_range: Tuple[int, int]  # MIDI note range (min, max)
    rhythm_complexity: float  # 0.0 to 1.0
    chord_distribution: Dict[str, float]  # Chord type probabilities
    scale_type: str  # "major", "minor", "modal", etc.
    articulation_density: float  # 0.0 to 1.0
    dynamic_range: Tuple[int, int]  # Velocity range (min, max)


class SonificationMapper:
    """
    Maps bio-vectors to musical parameters using deterministic rules.
    
    IMPORTANT: This module provides transparent conditioning signals.
    It does NOT claim any causal relationship between genes and music.
    The mappings serve as structured control signals for the generative model.
    """
    
    def __init__(self, 
                 tempo_range: Tuple[float, float] = (60.0, 180.0),
                 pitch_range: Tuple[int, int] = (36, 96),
                 key_mapping: str = "cycle_of_fifths",
                 chord_complexity_levels: int = 5):
        """
        Initialize the sonification mapper.
        
        Args:
            tempo_range: (min_tempo, max_tempo) in BPM
            pitch_range: (min_pitch, max_pitch) as MIDI notes
            key_mapping: Strategy for key selection
            chord_complexity_levels: Number of chord complexity levels
        """
        self.tempo_range = tempo_range
        self.pitch_range = pitch_range
        self.key_mapping = key_mapping
        self.chord_complexity_levels = chord_complexity_levels
        self.calibration = {}
        self.calibration_fitted = False
        
        # Define musical keys in cycle of fifths order
        self.keys_cycle = [
            "C_major", "G_major", "D_major", "A_major", "E_major", "B_major",
            "F#_major", "Db_major", "Ab_major", "Eb_major", "Bb_major", "F_major"
        ]
        
        # Relative minor keys
        self.minor_keys = {
            "C_major": "A_minor", "G_major": "E_minor", "D_major": "B_minor",
            "A_major": "F#_minor", "E_major": "C#_minor", "B_major": "G#_minor",
            "F#_major": "D#_minor", "Db_major": "Bb_minor", "Ab_major": "F_minor",
            "Eb_major": "C_minor", "Bb_major": "G_minor", "F_major": "D_minor"
        }
        
        # Basic chord distributions by complexity level
        self.chord_distributions = self._generate_chord_distributions()

    @staticmethod
    def _normalize(value: float, min_value: float, max_value: float) -> float:
        """Robust min-max normalization to [0, 1]."""
        denom = max(max_value - min_value, 1e-8)
        return float(np.clip((value - min_value) / denom, 0.0, 1.0))

    def fit_calibration(self, bio_vectors: np.ndarray,
                        lower_quantile: float = 0.05,
                        upper_quantile: float = 0.95) -> Dict[str, object]:
        """
        Fit data-driven calibration ranges from a batch of bio-vectors.

        This prevents collapse to a tiny subset of keys/tempi when raw features
        occupy narrow numeric ranges.
        """
        if bio_vectors is None or len(bio_vectors) == 0:
            self.calibration = {}
            self.calibration_fitted = False
            return self.calibration

        vectors = np.asarray(bio_vectors, dtype=np.float64)
        vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)

        nuc = vectors[:, :4] if vectors.shape[1] >= 4 else np.zeros((len(vectors), 4))
        entropy = vectors[:, 4] if vectors.shape[1] > 4 else np.ones(len(vectors))
        gc_skew = vectors[:, 5] if vectors.shape[1] > 5 else np.zeros(len(vectors))
        at_skew = vectors[:, 6] if vectors.shape[1] > 6 else np.zeros(len(vectors))
        gc_std = vectors[:, 92] if vectors.shape[1] > 92 else np.full(len(vectors), 0.1)
        entropy_std = vectors[:, 94] if vectors.shape[1] > 94 else np.full(len(vectors), 0.1)

        weights = np.array([0.0, 0.25, 0.5, 0.75], dtype=np.float64)
        key_projection = np.sum(nuc * weights, axis=1)
        quantile_grid = np.linspace(0.0, 1.0, len(self.keys_cycle) + 1)
        key_edges = np.quantile(key_projection, quantile_grid[1:-1])

        gc_content = nuc[:, 1] + nuc[:, 2]
        gc_thresholds = np.quantile(gc_content, [0.2, 0.4, 0.6, 0.8]).tolist()
        variability = (gc_std + entropy_std) / 2.0
        chord_signal = vectors[:, :16].mean(axis=1) if vectors.shape[1] >= 16 else np.zeros(len(vectors))
        combined_skew = (gc_skew + at_skew) / 2.0

        entropy_min = float(np.quantile(entropy, lower_quantile))
        entropy_max = float(np.quantile(entropy, upper_quantile))
        var_min = float(np.quantile(variability, lower_quantile))
        var_max = float(np.quantile(variability, upper_quantile))
        chord_min = float(np.quantile(chord_signal, lower_quantile))
        chord_max = float(np.quantile(chord_signal, upper_quantile))
        skew_scale = float(np.quantile(np.abs(combined_skew), 0.95))

        self.calibration = {
            'key_bin_edges': [float(x) for x in key_edges],
            'entropy_min': entropy_min,
            'entropy_max': entropy_max,
            'gc_thresholds': [float(x) for x in gc_thresholds],
            'articulation_min': var_min,
            'articulation_max': var_max,
            'chord_signal_min': chord_min,
            'chord_signal_max': chord_max,
            'skew_scale': max(skew_scale, 1e-3),
        }
        self.calibration_fitted = True
        return self.calibration

    def get_calibration_summary(self) -> Dict[str, object]:
        """Return calibration metadata for reporting."""
        if not self.calibration_fitted:
            return {'calibration_fitted': False}
        return {'calibration_fitted': True, **self.calibration}
    
    def _generate_chord_distributions(self) -> Dict[int, Dict[str, float]]:
        """Generate chord probability distributions for different complexity levels."""
        distributions = {}
        
        # Level 1: Only major and minor triads
        distributions[1] = {
            'major': 0.5, 'minor': 0.4, 'diminished': 0.1
        }
        
        # Level 2: Add seventh chords
        distributions[2] = {
            'major': 0.35, 'minor': 0.30, 'major7': 0.10, 'minor7': 0.10,
            'dominant7': 0.10, 'diminished': 0.05
        }
        
        # Level 3: More extended chords
        distributions[3] = {
            'major': 0.25, 'minor': 0.25, 'major7': 0.12, 'minor7': 0.12,
            'dominant7': 0.10, 'diminished': 0.06, 'augmented': 0.05,
            'sus4': 0.05
        }
        
        # Level 4: Jazz-influenced
        distributions[4] = {
            'major': 0.18, 'minor': 0.18, 'major7': 0.12, 'minor7': 0.12,
            'dominant7': 0.10, 'major9': 0.08, 'minor9': 0.08,
            'diminished': 0.05, 'augmented': 0.04, 'sus4': 0.05
        }
        
        # Level 5: Most complex
        distributions[5] = {
            'major': 0.12, 'minor': 0.12, 'major7': 0.10, 'minor7': 0.10,
            'dominant7': 0.10, 'major9': 0.10, 'minor9': 0.10,
            'dominant9': 0.08, 'diminished': 0.06, 'augmented': 0.05,
            'sus4': 0.04, 'altered': 0.03
        }
        
        return distributions
    
    def map_nucleotide_frequencies_to_key(self, nuc_freqs: np.ndarray) -> str:
        """
        Map nucleotide frequencies to musical key.
        
        Uses weighted combination of frequencies to select position in cycle of fifths.
        """
        if len(nuc_freqs) < 4:
            return "C_major"
        
        # Weighted sum based on nucleotide frequencies
        # A=0, C=1, G=2, T=3 mapping to cycle position
        weights = np.array([0.0, 0.25, 0.5, 0.75])
        position = np.sum(nuc_freqs * weights)

        if self.calibration_fitted and self.calibration.get('key_bin_edges'):
            edges = np.asarray(self.calibration['key_bin_edges'], dtype=np.float64)
            key_index = int(np.searchsorted(edges, position, side='right'))
            key_index = int(np.clip(key_index, 0, len(self.keys_cycle) - 1))
        else:
            # Fallback: map expected [0, 0.75] range to full key cycle.
            normalized = np.clip(position / 0.75, 0.0, 1.0)
            key_index = int(np.clip(round(normalized * (len(self.keys_cycle) - 1)), 0, len(self.keys_cycle) - 1))
        
        return self.keys_cycle[key_index]
    
    def map_entropy_to_tempo(self, entropy: float, min_entropy: float = 0.0, 
                             max_entropy: float = 2.0) -> float:
        """Map Shannon entropy to tempo."""
        if self.calibration_fitted:
            min_entropy = float(self.calibration.get('entropy_min', min_entropy))
            max_entropy = float(self.calibration.get('entropy_max', max_entropy))
        # Normalize entropy to [0, 1]
        norm_entropy = np.clip((entropy - min_entropy) / (max_entropy - min_entropy), 0.0, 1.0)
        
        # Higher entropy -> faster tempo (more activity)
        tempo = self.tempo_range[0] + norm_entropy * (self.tempo_range[1] - self.tempo_range[0])
        return round(tempo, 1)
    
    def map_skew_to_pitch_range(self, gc_skew: float, at_skew: float) -> Tuple[int, int]:
        """Map sequence skew statistics to pitch range."""
        # Combine skews
        combined_skew = (gc_skew + at_skew) / 2.0
        if self.calibration_fitted:
            skew_scale = float(self.calibration.get('skew_scale', 1.0))
            combined_skew = np.clip(combined_skew / max(skew_scale, 1e-6), -1.0, 1.0)
        else:
            # Clamp to [-1, 1]
            combined_skew = np.clip(combined_skew, -1.0, 1.0)
        
        # Map to pitch range shift
        base_min, base_max = self.pitch_range
        range_span = base_max - base_min
        
        # Negative skew -> lower range, positive skew -> higher range
        shift = int(combined_skew * (range_span / 4))
        
        new_min = max(12, base_min + shift)
        new_max = min(127, base_max + shift)
        
        return (new_min, new_max)
    
    def map_kmer_diversity_to_rhythm_complexity(self, kmer_dist: np.ndarray) -> float:
        """
        Map k-mer distribution diversity to rhythm complexity.
        
        Higher diversity -> more complex rhythms
        """
        if len(kmer_dist) == 0 or kmer_dist.sum() == 0:
            return 0.5
        
        # Compute normalized entropy of k-mer distribution
        probs = kmer_dist / kmer_dist.sum()
        probs = probs[probs > 0]  # Remove zeros for log
        diversity = -np.sum(probs * np.log2(probs))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(kmer_dist))
        if max_entropy == 0:
            return 0.5
        
        complexity = diversity / max_entropy
        return float(np.clip(complexity, 0.0, 1.0))
    
    def map_gc_content_to_scale_type(self, gc_content: float) -> str:
        """Map GC content to scale type."""
        gc_content = np.clip(gc_content, 0.0, 1.0)

        if self.calibration_fitted and self.calibration.get('gc_thresholds'):
            t1, t2, t3, t4 = self.calibration['gc_thresholds']
        else:
            t1, t2, t3, t4 = 0.35, 0.45, 0.55, 0.65

        if gc_content < t1:
            return "minor"
        elif gc_content < t2:
            return "dorian"
        elif gc_content < t3:
            return "major"
        elif gc_content < t4:
            return "mixolydian"
        else:
            return "lydian"
    
    def map_windowed_stats_to_articulation(self, gc_std: float, entropy_std: float) -> float:
        """Map variability in windowed statistics to articulation density."""
        # Higher variability -> more varied articulation
        variability = (gc_std + entropy_std) / 2.0
        if self.calibration_fitted:
            min_var = float(self.calibration.get('articulation_min', 0.0))
            max_var = float(self.calibration.get('articulation_max', 0.5))
            density = self._normalize(variability, min_var, max_var)
        else:
            density = np.clip(variability * 2.0, 0.0, 1.0)
        return float(density)
    
    def map_features_to_chord_distribution(self, bio_vector: np.ndarray) -> Dict[str, float]:
        """Map bio-vector features to chord distribution."""
        # Use first few dimensions to determine complexity level
        complexity_signal = np.mean(bio_vector[:16]) if len(bio_vector) >= 16 else 0.0

        # Normalize to complexity levels
        if self.calibration_fitted:
            min_signal = float(self.calibration.get('chord_signal_min', -1.0))
            max_signal = float(self.calibration.get('chord_signal_max', 1.0))
            normalized = self._normalize(complexity_signal, min_signal, max_signal)
        else:
            normalized = (complexity_signal + 1.0) / 2.0  # Assume centered features
        level = int(np.clip(normalized * self.chord_complexity_levels, 0, self.chord_complexity_levels - 1)) + 1
        
        return self.chord_distributions.get(level, self.chord_distributions[3])
    
    def bio_vector_to_musical_params(self, bio_vector: np.ndarray) -> MusicalParameters:
        """
        Convert complete bio-vector to musical parameters.
        
        Args:
            bio_vector: Fixed-dimensional bio-feature vector
            
        Returns:
            MusicalParameters dataclass instance
        """
        # Extract feature subsets (assumes standard ordering from extractor)
        # First 4: nucleotide frequencies
        nuc_freqs = bio_vector[:4] if len(bio_vector) >= 4 else np.zeros(4)
        
        # Next 1: entropy
        entropy = bio_vector[4] if len(bio_vector) > 4 else 1.0
        
        # Next 2: skews
        gc_skew = bio_vector[5] if len(bio_vector) > 5 else 0.0
        at_skew = bio_vector[6] if len(bio_vector) > 6 else 0.0
        
        # K-mer distributions start at index 7
        kmer_1 = bio_vector[7:11] if len(bio_vector) >= 11 else np.ones(4) / 4
        
        # Windowed stats (fixed positions from BioVectorExtractor ordering)
        gc_std = bio_vector[92] if len(bio_vector) > 92 else 0.1
        entropy_std = bio_vector[94] if len(bio_vector) > 94 else 0.1
        
        # Compute GC content from nucleotide frequencies
        gc_content = nuc_freqs[1] + nuc_freqs[2] if len(nuc_freqs) >= 3 else 0.5
        
        # Map to musical parameters
        key = self.map_nucleotide_frequencies_to_key(nuc_freqs)
        tempo = self.map_entropy_to_tempo(entropy)
        pitch_range = self.map_skew_to_pitch_range(gc_skew, at_skew)
        rhythm_complexity = self.map_kmer_diversity_to_rhythm_complexity(kmer_1)
        scale_type = self.map_gc_content_to_scale_type(gc_content)
        articulation_density = self.map_windowed_stats_to_articulation(gc_std, entropy_std)
        chord_distribution = self.map_features_to_chord_distribution(bio_vector)
        
        # Dynamic range based on rhythm complexity
        dynamic_min = max(20, 80 - int(rhythm_complexity * 40))
        dynamic_max = min(127, 90 + int(rhythm_complexity * 30))
        
        return MusicalParameters(
            key=key,
            tempo=tempo,
            pitch_range=pitch_range,
            rhythm_complexity=rhythm_complexity,
            chord_distribution=chord_distribution,
            scale_type=scale_type,
            articulation_density=articulation_density,
            dynamic_range=(dynamic_min, dynamic_max)
        )
    
    def create_conditioning_vector(self, musical_params: MusicalParameters) -> np.ndarray:
        """
        Create a conditioning vector from musical parameters for model input.
        
        This creates a continuous vector representation suitable for neural network conditioning.
        """
        # Encode key as one-hot
        key_idx = self.keys_cycle.index(musical_params.key) if musical_params.key in self.keys_cycle else 0
        key_onehot = np.zeros(len(self.keys_cycle))
        key_onehot[key_idx] = 1.0
        
        # Normalize tempo
        tempo_norm = np.clip(
            (musical_params.tempo - self.tempo_range[0]) / (self.tempo_range[1] - self.tempo_range[0]),
            0.0,
            1.0
        )
        
        # Normalize pitch range
        pitch_min_norm = musical_params.pitch_range[0] / 127.0
        pitch_max_norm = musical_params.pitch_range[1] / 127.0
        
        # Encode scale type
        scale_types = ["major", "minor", "dorian", "mixolydian", "lydian"]
        scale_idx = scale_types.index(musical_params.scale_type) if musical_params.scale_type in scale_types else 0
        scale_onehot = np.zeros(len(scale_types))
        scale_onehot[scale_idx] = 1.0
        
        # Other parameters (already normalized)
        other_params = np.array([
            musical_params.rhythm_complexity,
            musical_params.articulation_density,
            musical_params.dynamic_range[0] / 127.0,
            musical_params.dynamic_range[1] / 127.0
        ])
        
        # Concatenate all components
        conditioning = np.concatenate([
            key_onehot,
            np.array([tempo_norm, pitch_min_norm, pitch_max_norm]),
            scale_onehot,
            other_params
        ])
        
        return conditioning


def apply_sonification_rules(bio_vectors: np.ndarray) -> np.ndarray:
    """
    Apply sonification rules to batch of bio-vectors.
    
    Args:
        bio_vectors: Array of shape (n_samples, bio_vector_dim)
        
    Returns:
        Array of conditioning vectors of shape (n_samples, conditioning_dim)
    """
    mapper = SonificationMapper()
    mapper.fit_calibration(bio_vectors)
    conditioning_vectors = []
    
    for bio_vec in bio_vectors:
        musical_params = mapper.bio_vector_to_musical_params(bio_vec)
        cond_vec = mapper.create_conditioning_vector(musical_params)
        conditioning_vectors.append(cond_vec)
    
    return np.array(conditioning_vectors)
