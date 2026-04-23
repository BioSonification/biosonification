"""
Diversity Analysis for Bio-Music Pipeline.

Analyzes whether the model generates diverse music for different bio-vectors
or whether all outputs sound the same.

Metrics:
- Pairwise distance between generated sequences
- Intra-bio-vector vs inter-bio-vector variance
- Vocabulary coverage
- Novelty vs training data
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from collections import Counter


class DiversityAnalyzer:
    """
    Analyze diversity of generated music across different bio-vectors.
    """

    def __init__(self, vocab: Dict[str, int] = None):
        """
        Initialize diversity analyzer.

        Args:
            vocab: Token vocabulary
        """
        self.vocab = vocab

    def compute_pairwise_distances(self,
                                    sequences: np.ndarray,
                                    metric: str = 'edit') -> np.ndarray:
        """
        Compute pairwise distances between sequences.

        Args:
            sequences: Array of shape (n_sequences, seq_len)
            metric: Distance metric ('edit', 'jaccard', 'cosine')

        Returns:
            Distance matrix of shape (n_sequences, n_sequences)
        """
        n = len(sequences)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                if metric == 'edit':
                    dist = self._edit_distance(sequences[i], sequences[j])
                elif metric == 'jaccard':
                    dist = 1.0 - self._jaccard_similarity(sequences[i], sequences[j])
                elif metric == 'cosine':
                    dist = 1.0 - self._cosine_similarity(sequences[i], sequences[j])
                else:
                    raise ValueError(f"Unknown metric: {metric}")

                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def _edit_distance(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute normalized edit distance between two sequences."""
        # Simple character-level edit distance on token sequences
        s1, s2 = seq1.tolist(), seq2.tolist()
        m, n = len(s1), len(s2)

        # Use dynamic programming (optimized for short sequences)
        if m * n > 100000:  # Too large, approximate
            return abs(m - n) / max(m, n)

        dp = np.zeros((m + 1, n + 1))
        for i in range(m + 1):
            dp[i, 0] = i
        for j in range(n + 1):
            dp[0, j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                dp[i, j] = min(
                    dp[i-1, j] + 1,      # deletion
                    dp[i, j-1] + 1,      # insertion
                    dp[i-1, j-1] + cost   # substitution
                )

        return dp[m, n] / max(m, n)

    def _jaccard_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute Jaccard similarity of token sets."""
        set1 = set(seq1.tolist())
        set2 = set(seq2.tolist())
        intersection = set1 & set2
        union = set1 | set2
        return len(intersection) / max(len(union), 1)

    def _cosine_similarity(self, seq1: np.ndarray, seq2: np.ndarray) -> float:
        """Compute cosine similarity of token distributions."""
        # Use token frequency vectors
        counter1 = Counter(seq1.tolist())
        counter2 = Counter(seq2.tolist())
        all_tokens = set(counter1.keys()) | set(counter2.keys())

        vec1 = np.array([counter1.get(t, 0) for t in all_tokens])
        vec2 = np.array([counter2.get(t, 0) for t in all_tokens])

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def compute_intra_inter_variance(self,
                                      sequences_by_bio: Dict[str, np.ndarray],
                                      metric: str = 'edit') -> Dict:
        """
        Compare variance within same bio-vector vs across different bio-vectors.

        Args:
            sequences_by_bio: Dict mapping bio-vector ID to multiple generated sequences
            metric: Distance metric

        Returns:
            Dictionary with variance analysis
        """
        # Intra-bio-vector distances
        intra_distances = []
        for bio_id, seqs in sequences_by_bio.items():
            if len(seqs) > 1:
                dists = self.compute_pairwise_distances(seqs, metric)
                # Upper triangle
                upper_tri = dists[np.triu_indices_from(dists, k=1)]
                intra_distances.extend(upper_tri.tolist())

        # Inter-bio-vector distances
        inter_distances = []
        bio_ids = list(sequences_by_bio.keys())
        for i in range(len(bio_ids)):
            for j in range(i + 1, len(bio_ids)):
                # Sample one sequence from each
                seq1 = sequences_by_bio[bio_ids[i]][0]
                seq2 = sequences_by_bio[bio_ids[j]][0]

                if metric == 'edit':
                    dist = self._edit_distance(seq1, seq2)
                elif metric == 'jaccard':
                    dist = 1.0 - self._jaccard_similarity(seq1, seq2)
                else:
                    dist = 1.0 - self._cosine_similarity(seq1, seq2)

                inter_distances.append(dist)

        # Statistical test
        from scipy import stats
        if intra_distances and inter_distances:
            t_stat, p_value = stats.ttest_ind(intra_distances, inter_distances)
            effect_size = (np.mean(inter_distances) - np.mean(intra_distances)) / max(
                np.sqrt((np.var(intra_distances) + np.var(inter_distances)) / 2), 1e-10
            )
        else:
            t_stat, p_value, effect_size = 0, 1.0, 0

        return {
            'intra_bio_mean': float(np.mean(intra_distances)) if intra_distances else 0,
            'intra_bio_std': float(np.std(intra_distances)) if intra_distances else 0,
            'inter_bio_mean': float(np.mean(inter_distances)) if inter_distances else 0,
            'inter_bio_std': float(np.std(inter_distances)) if inter_distances else 0,
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'effect_size_cohens_d': float(effect_size),
            'significant_difference': bool(p_value < 0.05),
            'n_intra_comparisons': len(intra_distances),
            'n_inter_comparisons': len(inter_distances)
        }

    def compute_vocabulary_coverage(self,
                                     sequences: np.ndarray,
                                     vocab_size: int = None) -> Dict:
        """
        Compute vocabulary coverage of generated sequences.

        Args:
            sequences: Array of generated sequences
            vocab_size: Total vocabulary size

        Returns:
            Dictionary with coverage statistics
        """
        all_tokens = set()
        coverages = []

        for seq in sequences:
            seq_tokens = set(seq.tolist())
            all_tokens |= seq_tokens
            if vocab_size:
                coverages.append(len(seq_tokens) / vocab_size)

        if vocab_size:
            overall_coverage = len(all_tokens) / vocab_size
        else:
            overall_coverage = 1.0
            coverages = [1.0] * len(sequences)

        return {
            'overall_coverage': float(overall_coverage),
            'unique_tokens_used': len(all_tokens),
            'total_vocab_size': vocab_size or len(all_tokens),
            'mean_per_sequence_coverage': float(np.mean(coverages)) if coverages else 0,
            'std_per_sequence_coverage': float(np.std(coverages)) if coverages else 0
        }

    def compute_novelty_vs_training(self,
                                     generated: np.ndarray,
                                     training: np.ndarray,
                                     metric: str = 'edit') -> Dict:
        """
        Compute how novel generated sequences are compared to training data.

        Args:
            generated: Generated sequences
            training: Training sequences
            metric: Distance metric

        Returns:
            Dictionary with novelty metrics
        """
        # For each generated sequence, find distance to nearest training sequence
        min_distances = []

        for gen_seq in generated[:50]:  # Sample for efficiency
            min_dist = float('inf')
            for train_seq in training:
                if metric == 'edit':
                    dist = self._edit_distance(gen_seq, train_seq)
                elif metric == 'jaccard':
                    dist = 1.0 - self._jaccard_similarity(gen_seq, train_seq)
                else:
                    dist = 1.0 - self._cosine_similarity(gen_seq, train_seq)

                min_dist = min(min_dist, dist)

            min_distances.append(min_dist)

        return {
            'mean_novelty': float(np.mean(min_distances)),
            'min_novelty': float(np.min(min_distances)),
            'max_novelty': float(np.max(min_distances)),
            'std_novelty': float(np.std(min_distances)),
            'fraction_novel_gt_05': float(np.mean([d > 0.5 for d in min_distances]))
        }

    def analyze_full_diversity(self,
                                generated_samples: Dict[str, np.ndarray],
                                training_data: np.ndarray = None,
                                vocab_size: int = None) -> Dict:
        """
        Run complete diversity analysis.

        Args:
            generated_samples: Dict mapping condition to sequences
            training_data: Optional training sequences for novelty analysis
            vocab_size: Total vocabulary size

        Returns:
            Complete diversity analysis results
        """
        results = {}

        # 1. Vocabulary coverage per condition
        print("Computing vocabulary coverage...")
        for condition, sequences in generated_samples.items():
            results[f'{condition}_vocab_coverage'] = self.compute_vocabulary_coverage(
                sequences, vocab_size
            )

        # 2. Pairwise distances per condition
        print("Computing pairwise distances...")
        for condition, sequences in generated_samples.items():
            if len(sequences) > 1:
                distances = self.compute_pairwise_distances(
                    sequences[:25], metric='jaccard'  # Limit for speed
                )
                upper_tri = distances[np.triu_indices_from(distances, k=1)]
                results[f'{condition}_pairwise_distance'] = {
                    'mean': float(np.mean(upper_tri)),
                    'std': float(np.std(upper_tri)),
                    'min': float(np.min(upper_tri)),
                    'max': float(np.max(upper_tri))
                }

        # 3. Intra vs inter bio-vector variance (if multiple samples per bio)
        if 'conditioned' in generated_samples:
            print("Computing intra/inter bio-vector variance...")
            # Treat each sequence as from different bio-vector (simplified)
            n_bios = min(10, len(generated_samples['conditioned']) // 3)
            sequences_by_bio = {}
            for i in range(n_bios):
                sequences_by_bio[f'bio_{i}'] = generated_samples['conditioned'][i*3:(i+1)*3]

            if len(sequences_by_bio) > 1:
                results['intra_inter_variance'] = self.compute_intra_inter_variance(
                    sequences_by_bio, metric='jaccard'
                )

        # 4. Novelty vs training data
        if training_data is not None:
            print("Computing novelty vs training data...")
            for condition, sequences in generated_samples.items():
                results[f'{condition}_novelty'] = self.compute_novelty_vs_training(
                    sequences[:50], training_data, metric='jaccard'
                )

        return results
