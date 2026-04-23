"""
IDyOM Integration Module.

Wrapper around IDyOMpy (Information Dynamics of Music, Marcus Pearce)
for computing musical complexity metrics.

IMPORTANT LIMITATION:
IDyOM only supports MONOPHONIC MIDI files. Polyphonic MIDI (like MAESTRO
piano recordings) will raise RuntimeError. For polyphonic files, use
ShannonEntropyMetrics from perplexity_metrics.py instead.

This module:
1. Detects if MIDI is monophonic or polyphonic
2. Runs IDyOM on monophonic files
3. Falls back to Shannon entropy for polyphonic files
4. Provides comparison between IDyOM and Shannon entropy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import sys


class IDyOMWrapper:
    """
    Wrapper for IDyOM model that handles both monophonic and polyphonic MIDI.

    For monophonic MIDI: uses IDyOM (Information Content + Entropy)
    For polyphonic MIDI: falls back to Shannon entropy (pitch class based)
    """

    def __init__(self, idyom_path: str = None, max_order: int = 20,
                 quantization: int = 24, view_points: List[str] = None):
        """
        Initialize IDyOM wrapper.

        Args:
            idyom_path: Path to IDyOMpy repository
            max_order: Maximum Markov order for IDyOM model
            quantization: Rhythmic quantization (ticks per beat)
            view_points: IDyOM viewpoints to use (default: ['pitch'])
        """
        if idyom_path is None:
            # Default path relative to project root
            project_root = Path(__file__).parent.parent.parent
            idyom_path = str(project_root / "tools" / "IDyOMpy")

        # Add IDyOMpy to path
        if idyom_path not in sys.path:
            sys.path.insert(0, idyom_path)

        try:
            from idyom.idyom import idyom as IDyOMModel
            from idyom.data import data as IDyOMData
            self._idyom_model = IDyOMModel
            self._idyom_data = IDyOMData
            self._idyom_available = True
        except (ImportError, ModuleNotFoundError) as e:
            print(f"Warning: IDyOMpy not available: {e}")
            self._idyom_available = False
            self._idyom_model = None
            self._idyom_data = None

        self.max_order = max_order
        self.quantization = quantization
        self.view_points = view_points or ['pitch']

    def is_monophonic(self, midi_path: str) -> bool:
        """
        Check if MIDI file is monophonic (single voice).

        A MIDI is considered monophonic if at any point in time,
        there is at most one active note.
        """
        import mido

        try:
            midi = mido.MidiFile(midi_path)
        except Exception:
            return False

        # Merge all tracks and check for simultaneous notes
        active_notes = 0
        max_simultaneous = 0

        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes += 1
                    max_simultaneous = max(max_simultaneous, active_notes)
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    active_notes = max(0, active_notes - 1)

        return max_simultaneous <= 1

    def compute_idyom_entropy(self, midi_path: str) -> Optional[Dict]:
        """
        Compute IDyOM-based complexity for a MONOPHONIC MIDI file.

        Returns:
            Dictionary with:
            - mean_ic: Mean Information Content per note
            - mean_entropy: Mean relative entropy per note
            - total_ic: Total Information Content
            - std_ic: Standard deviation of IC
            - std_entropy: Standard deviation of entropy
            - n_notes: Number of notes
            - method: 'idyom'
        """
        if not self._idyom_available:
            return None

        try:
            L = self._idyom_model(maxOrder=self.max_order, viewPoints=self.view_points)
            M = self._idyom_data(quantization=self.quantization)
            M.addFiles([midi_path])
            L.train(M)

            IC, E = L.getSurprisefromFile(midi_path)

            return {
                'mean_ic': float(np.mean(IC)),
                'mean_entropy': float(np.mean(E)),
                'total_ic': float(np.sum(IC)),
                'std_ic': float(np.std(IC)),
                'std_entropy': float(np.std(E)),
                'n_notes': len(IC),
                'method': 'idyom'
            }
        except RuntimeError as e:
            if "polyphonic" in str(e).lower():
                return None
            raise
        except Exception as e:
            print(f"Warning: IDyOM failed for {midi_path}: {e}")
            return None

    def compute_shannon_fallback(self, midi_path: str) -> Dict:
        """
        Compute Shannon entropy as fallback for polyphonic MIDI.

        Returns:
            Dictionary with pitch entropy metrics
        """
        import mido
        from collections import Counter

        try:
            midi = mido.MidiFile(midi_path)
        except Exception:
            return {'mean_pitch_entropy': 0.0, 'method': 'shannon_fallback', 'n_notes': 0}

        pitch_classes = []
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    pitch_classes.append(msg.note % 12)

        if not pitch_classes:
            return {'mean_pitch_entropy': 0.0, 'method': 'shannon_fallback', 'n_notes': 0}

        total = len(pitch_classes)
        pc_counts = Counter(pitch_classes)
        probs = [count / total for count in pc_counts.values()]
        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(12)

        return {
            'mean_pitch_entropy': float(entropy / max_entropy),
            'method': 'shannon_fallback',
            'n_notes': total
        }

    def compute_complexity(self, midi_path: str) -> Dict:
        """
        Compute complexity using IDyOM if monophonic, else Shannon entropy.

        Returns:
            Dictionary with all metrics + complexity score
        """
        # Try IDyOM first
        if self._idyom_available:
            idyom_result = self.compute_idyom_entropy(midi_path)
            if idyom_result is not None:
                idyom_result['is_monophonic'] = True
                # Normalize IC to [0, 1] (IC is in bits, max ~log2(128) for 128 pitches)
                max_ic = np.log2(128)  # ~7 bits for MIDI
                idyom_result['normalized_ic'] = min(idyom_result['mean_ic'] / max_ic, 1.0)
                idyom_result['complexity_score'] = idyom_result['normalized_ic']
                return idyom_result

        # Fallback to Shannon
        shannon_result = self.compute_shannon_fallback(midi_path)
        shannon_result['is_monophonic'] = self.is_monophonic(midi_path)
        shannon_result['complexity_score'] = shannon_result['mean_pitch_entropy']
        return shannon_result


def compare_idyom_vs_shannon(midi_files: List[str],
                              output_path: str = None) -> Dict:
    """
    Compare IDyOM entropy vs Shannon entropy across MIDI files.

    For monophonic files: computes BOTH IDyOM and Shannon entropy
    For polyphonic files: computes Shannon entropy only

    Args:
        midi_files: List of MIDI file paths
        output_path: Optional path to save comparison CSV

    Returns:
        Dictionary with comparison results
    """
    from tqdm import tqdm
    from collections import Counter
    import mido

    wrapper = IDyOMWrapper()
    results = {}

    monophonic_count = 0
    polyphonic_count = 0

    for midi_path in tqdm(midi_files, desc="Computing IDyOM vs Shannon"):
        is_mono = wrapper.is_monophonic(midi_path)

        if is_mono:
            monophonic_count += 1
            # Compute BOTH IDyOM and Shannon
            idyom_result = wrapper.compute_idyom_entropy(midi_path)
            shannon_result = wrapper.compute_shannon_fallback(midi_path)

            if idyom_result is not None:
                results[midi_path] = {
                    'is_monophonic': True,
                    'idyom_mean_ic': idyom_result['mean_ic'],
                    'idyom_mean_entropy': idyom_result['mean_entropy'],
                    'idyom_total_ic': idyom_result['total_ic'],
                    'shannon_pitch_entropy': shannon_result['mean_pitch_entropy'],
                    'n_notes': idyom_result['n_notes'],
                }
            else:
                results[midi_path] = {
                    'is_monophonic': True,
                    'idyom_mean_ic': None,
                    'idyom_mean_entropy': None,
                    'idyom_total_ic': None,
                    'shannon_pitch_entropy': shannon_result['mean_pitch_entropy'],
                    'n_notes': shannon_result['n_notes'],
                }
        else:
            polyphonic_count += 1
            # Shannon only
            shannon_result = wrapper.compute_shannon_fallback(midi_path)
            results[midi_path] = {
                'is_monophonic': False,
                'idyom_mean_ic': None,
                'idyom_mean_entropy': None,
                'idyom_total_ic': None,
                'shannon_pitch_entropy': shannon_result['mean_pitch_entropy'],
                'n_notes': shannon_result['n_notes'],
            }

    # Compute correlation for monophonic files
    mono_files = {k: v for k, v in results.items() if v['is_monophonic'] and v['idyom_mean_ic'] is not None}

    correlation_results = {}
    if len(mono_files) > 2:
        idyom_ics = [v['idyom_mean_ic'] for v in mono_files.values()]
        shannon_ents = [v['shannon_pitch_entropy'] for v in mono_files.values()]

        from scipy import stats
        pearson_r, pearson_p = stats.pearsonr(idyom_ics, shannon_ents)
        spearman_r, spearman_p = stats.spearmanr(idyom_ics, shannon_ents)

        correlation_results = {
            'n_monophonic_files': len(mono_files),
            'n_polyphonic_files': polyphonic_count,
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'idyom_ic_mean': float(np.mean(idyom_ics)),
            'idyom_ic_std': float(np.std(idyom_ics)),
            'shannon_entropy_mean': float(np.mean(shannon_ents)),
            'shannon_entropy_std': float(np.std(shannon_ents)),
        }

    # Save CSV if output path provided
    if output_path:
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'midi_path', 'is_monophonic',
                'idyom_mean_ic', 'idyom_mean_entropy', 'idyom_total_ic',
                'shannon_pitch_entropy', 'n_notes'
            ])
            for path, metrics in results.items():
                writer.writerow([
                    path,
                    metrics['is_monophonic'],
                    f"{metrics['idyom_mean_ic']:.4f}" if metrics['idyom_mean_ic'] is not None else 'N/A',
                    f"{metrics['idyom_mean_entropy']:.4f}" if metrics['idyom_mean_entropy'] is not None else 'N/A',
                    f"{metrics['idyom_total_ic']:.4f}" if metrics['idyom_total_ic'] is not None else 'N/A',
                    f"{metrics['shannon_pitch_entropy']:.4f}",
                    metrics['n_notes']
                ])
        print(f"Comparison results saved to: {output_path}")

    return {
        'per_file': results,
        'correlation': correlation_results
    }


def plot_idyom_vs_shannon(correlation_results: Dict,
                            output_path: str = None):
    """
    Create scatter plot: IDyOM IC vs Shannon Entropy.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    per_file = correlation_results.get('per_file', {})
    mono_files = {k: v for k, v in per_file.items()
                  if v['is_monophonic'] and v['idyom_mean_ic'] is not None}

    if len(mono_files) < 2:
        print("Not enough monophonic files for scatter plot")
        return

    idyom_ics = [v['idyom_mean_ic'] for v in mono_files.values()]
    shannon_ents = [v['shannon_pitch_entropy'] for v in mono_files.values()]

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(idyom_ics, shannon_ents, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Add regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(idyom_ics, shannon_ents)
    x_line = np.linspace(min(idyom_ics), max(idyom_ics), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, 'r--', alpha=0.8,
            label=f'r = {r_value:.3f}, p = {p_value:.4f}')

    corr = correlation_results.get('correlation', {})
    ax.set_xlabel(f'IDyOM Mean Information Content (bits)\n'
                  f'mean = {corr.get("idyom_ic_mean", 0):.3f}, '
                  f'std = {corr.get("idyom_ic_std", 0):.3f}', fontsize=12)
    ax.set_ylabel(f'Shannon Pitch Entropy (normalized)\n'
                  f'mean = {corr.get("shannon_entropy_mean", 0):.3f}, '
                  f'std = {corr.get("shannon_entropy_std", 0):.3f}', fontsize=12)
    ax.set_title('IDyOM vs Shannon Entropy\n'
                 f'Pearson r = {corr.get("pearson_r", 0):.3f} '
                 f'(p = {corr.get("pearson_p", 0):.4f})\n'
                 f'n = {corr.get("n_monophonic_files", 0)} monophonic files',
                 fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Scatter plot saved to: {output_path}")

    plt.close(fig)
