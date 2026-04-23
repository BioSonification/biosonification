"""
Stage 1: Bio-vector extraction from genomic sequences.

This module extracts fixed-dimensional feature vectors from biological sequences.
Features include nucleotide frequencies, entropy, k-mer distributions, and shift statistics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import re


class BioVectorExtractor:
    """
    Extracts fixed-dimensional bio-vectors from genomic sequence files.
    
    Features computed:
    - Nucleotide frequencies (A, C, G, T)
    - Shannon entropy
    - K-mer distributions for multiple k values
    - Shift statistics (GC skew, AT skew)
    """
    
    def __init__(self, kmer_sizes: List[int] = [1, 2, 3], 
                 window_size: int = 1000, 
                 stride: int = 500,
                 min_sequence_length: int = 100):
        """
        Initialize the extractor with configuration parameters.
        
        Args:
            kmer_sizes: List of k values for k-mer computation
            window_size: Size of sliding window for local statistics
            stride: Stride for sliding window
            min_sequence_length: Minimum valid sequence length
        """
        self.kmer_sizes = kmer_sizes
        self.window_size = window_size
        self.stride = stride
        self.min_sequence_length = min_sequence_length
        self.nucleotides = ['A', 'C', 'G', 'T']
        
    def read_fasta(self, filepath: str) -> List[Tuple[str, str]]:
        """
        Read sequences from FASTA file.
        
        Args:
            filepath: Path to FASTA file
            
        Returns:
            List of (header, sequence) tuples
        """
        sequences = []
        current_header = None
        current_seq = []
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_header is not None:
                            seq = ''.join(current_seq).upper()
                            seq = re.sub(r'[^ACGT]', '', seq)  # Remove non-standard bases
                            sequences.append((current_header, seq))
                        current_header = line[1:]
                        current_seq = []
                    else:
                        current_seq.append(line)
                
                # Don't forget the last sequence
                if current_header is not None:
                    seq = ''.join(current_seq).upper()
                    seq = re.sub(r'[^ACGT]', '', seq)
                    sequences.append((current_header, seq))
        except FileNotFoundError:
            raise FileNotFoundError(f"FASTA file not found: {filepath}")
        except Exception as e:
            raise IOError(f"Error reading FASTA file: {e}")
        
        return sequences
    
    def compute_nucleotide_frequencies(self, sequence: str) -> np.ndarray:
        """Compute normalized nucleotide frequencies."""
        counts = Counter(sequence)
        total = len(sequence)
        if total == 0:
            return np.zeros(4)
        freqs = np.array([counts.get(nuc, 0) / total for nuc in self.nucleotides])
        return freqs
    
    def compute_shannon_entropy(self, sequence: str) -> float:
        """Compute Shannon entropy of the sequence."""
        if len(sequence) == 0:
            return 0.0
        counts = Counter(sequence)
        total = len(sequence)
        entropy = 0.0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        return entropy
    
    def compute_kmer_distribution(self, sequence: str, k: int) -> np.ndarray:
        """
        Compute normalized k-mer distribution.
        
        Args:
            sequence: DNA sequence
            k: Length of k-mers
            
        Returns:
            Normalized k-mer frequency vector
        """
        if len(sequence) < k:
            return np.zeros(4 ** k)
        
        # Generate all possible k-mers
        kmer_counts = Counter()
        for i in range(len(sequence) - k + 1):
            kmer = sequence[i:i+k]
            if all(nuc in self.nucleotides for nuc in kmer):
                kmer_counts[kmer] += 1
        
        # Create fixed-size vector
        kmer_list = []
        def generate_kmers(prefix: str, length: int):
            if length == 0:
                kmer_list.append(prefix)
                return
            for nuc in self.nucleotides:
                generate_kmers(prefix + nuc, length - 1)
        
        generate_kmers('', k)
        
        total = sum(kmer_counts.values())
        if total == 0:
            return np.zeros(4 ** k)
        
        distribution = np.array([kmer_counts.get(kmer, 0) / total for kmer in kmer_list])
        return distribution
    
    def compute_gc_skew(self, sequence: str) -> float:
        """Compute GC skew: (G - C) / (G + C)."""
        counts = Counter(sequence)
        g_count = counts.get('G', 0)
        c_count = counts.get('C', 0)
        total = g_count + c_count
        if total == 0:
            return 0.0
        return (g_count - c_count) / total
    
    def compute_at_skew(self, sequence: str) -> float:
        """Compute AT skew: (A - T) / (A + T)."""
        counts = Counter(sequence)
        a_count = counts.get('A', 0)
        t_count = counts.get('T', 0)
        total = a_count + t_count
        if total == 0:
            return 0.0
        return (a_count - t_count) / total
    
    def compute_windowed_statistics(self, sequence: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute windowed statistics across the sequence.
        
        Returns:
            Tuple of (gc_content_windows, entropy_windows)
        """
        if len(sequence) < self.window_size:
            # Use whole sequence if shorter than window
            gc_content = np.array([(sequence.count('G') + sequence.count('C')) / max(len(sequence), 1)])
            entropy = np.array([self.compute_shannon_entropy(sequence)])
            return gc_content, entropy
        
        n_windows = (len(sequence) - self.window_size) // self.stride + 1
        gc_contents = np.zeros(n_windows)
        entropies = np.zeros(n_windows)
        
        for i in range(n_windows):
            start = i * self.stride
            end = start + self.window_size
            window = sequence[start:end]
            
            gc_contents[i] = (window.count('G') + window.count('C')) / self.window_size
            entropies[i] = self.compute_shannon_entropy(window)
        
        return gc_contents, entropies
    
    def extract_features(self, sequence: str) -> Dict[str, np.ndarray]:
        """
        Extract all features from a single sequence.
        
        Args:
            sequence: DNA sequence string
            
        Returns:
            Dictionary of feature arrays
        """
        if len(sequence) < self.min_sequence_length:
            raise ValueError(f"Sequence too short: {len(sequence)} < {self.min_sequence_length}")
        
        features = {}
        
        # Global statistics
        features['nuc_freqs'] = self.compute_nucleotide_frequencies(sequence)
        features['entropy'] = np.array([self.compute_shannon_entropy(sequence)])
        features['gc_skew'] = np.array([self.compute_gc_skew(sequence)])
        features['at_skew'] = np.array([self.compute_at_skew(sequence)])
        
        # K-mer distributions
        for k in self.kmer_sizes:
            features[f'kmer_{k}'] = self.compute_kmer_distribution(sequence, k)
        
        # Windowed statistics
        gc_windows, entropy_windows = self.compute_windowed_statistics(sequence)
        features['gc_window_mean'] = np.array([gc_windows.mean()])
        features['gc_window_std'] = np.array([gc_windows.std()])
        features['entropy_window_mean'] = np.array([entropy_windows.mean()])
        features['entropy_window_std'] = np.array([entropy_windows.std()])
        
        return features
    
    def create_bio_vector(self, features: Dict[str, np.ndarray], target_dim: int = 128) -> np.ndarray:
        """
        Concatenate and project features to fixed-dimensional bio-vector.
        
        Args:
            features: Dictionary of feature arrays
            target_dim: Target dimension for bio-vector
            
        Returns:
            Fixed-dimensional bio-vector
        """
        # Fixed feature order to keep indices stable across mapping/ablation code.
        ordered_keys = [
            'nuc_freqs',
            'entropy',
            'gc_skew',
            'at_skew',
            'kmer_1',
            'kmer_2',
            'kmer_3',
            'gc_window_mean',
            'gc_window_std',
            'entropy_window_mean',
            'entropy_window_std',
        ]

        core_parts = []
        for key in ordered_keys:
            value = features.get(key)
            if value is None:
                continue
            core_parts.append(value.flatten())

        if core_parts:
            core = np.concatenate(core_parts)
        else:
            core = np.zeros(0, dtype=np.float32)

        # Reserve last 4 dims for summary statistics to match ablation indices.
        main_dim = max(target_dim - 4, 1)
        vector = np.zeros(main_dim, dtype=np.float32)
        copy_len = min(len(core), main_dim)
        if copy_len > 0:
            vector[:copy_len] = core[:copy_len]

        source_for_stats = core if len(core) > 0 else np.zeros(1, dtype=np.float32)
        summary_stats = np.array([
            source_for_stats.mean(),
            source_for_stats.std(),
            source_for_stats.min(),
            source_for_stats.max()
        ], dtype=np.float32)

        result = np.concatenate([vector, summary_stats])
        if len(result) > target_dim:
            result = result[:target_dim]
        elif len(result) < target_dim:
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[:len(result)] = result
            result = padded
        return result
    
    def process_file(self, filepath: str, target_dim: int = 128) -> List[Tuple[str, np.ndarray]]:
        """
        Process entire FASTA file and return bio-vectors.
        
        Args:
            filepath: Path to FASTA file
            target_dim: Target dimension for bio-vectors
            
        Returns:
            List of (sequence_id, bio_vector) tuples
        """
        sequences = self.read_fasta(filepath)
        results = []
        
        for header, sequence in sequences:
            if len(sequence) >= self.min_sequence_length:
                try:
                    features = self.extract_features(sequence)
                    bio_vector = self.create_bio_vector(features, target_dim)
                    results.append((header, bio_vector))
                except Exception as e:
                    print(f"Warning: Failed to process sequence {header}: {e}")
        
        return results


def extract_bio_vectors_from_sequences(sequences: List[str], 
                                       target_dim: int = 128,
                                       **kwargs) -> np.ndarray:
    """
    Convenience function to extract bio-vectors from list of sequences.
    
    Args:
        sequences: List of DNA sequence strings
        target_dim: Target dimension for bio-vectors
        **kwargs: Additional arguments for BioVectorExtractor
        
    Returns:
        Array of shape (n_sequences, target_dim)
    """
    extractor = BioVectorExtractor(**kwargs)
    vectors = []
    
    for i, seq in enumerate(sequences):
        seq_clean = re.sub(r'[^ACGT]', '', seq.upper())
        if len(seq_clean) >= extractor.min_sequence_length:
            features = extractor.extract_features(seq_clean)
            vector = extractor.create_bio_vector(features, target_dim)
            vectors.append(vector)
        else:
            # Create zero vector for invalid sequences
            vectors.append(np.zeros(target_dim))
    
    return np.array(vectors)
