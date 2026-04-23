"""
Paired Dataset Creator: Creates MIDI ↔ bio-vector pairs for training.

This module assigns each MIDI file from MAESTRO dataset a corresponding
bio-vector from real genomic sequences, creating proper training pairs
instead of random assignment.

Strategy:
1. Load real FASTA sequences (Homo sapiens CDS)
2. Extract bio-vectors from each sequence
3. Assign bio-vectors to MIDI files based on musical characteristics:
   - Compute musical features from MIDI (pitch stats, tempo, density)
   - Sort both MIDI and bio-vectors by complexity
   - Match by rank: simple MIDI ↔ simple bio-vector, complex ↔ complex
4. Save paired data for training
"""

import json
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import Counter
import mido
from tqdm import tqdm

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from bio_music_pipeline import set_seed
from bio_music_pipeline.extractors import BioVectorExtractor, FastaDatasetLoader
from bio_music_pipeline.sonification import SonificationMapper
from bio_music_pipeline.data import MIDIPreprocessor


@dataclass
class MIDIFeatures:
    """Statistical features extracted from a MIDI file."""
    filepath: str
    mean_pitch: float = 0.0
    std_pitch: float = 0.0
    note_density: float = 0.0  # notes per second
    tempo: float = 120.0
    pitch_range_span: float = 0.0
    velocity_mean: float = 0.0
    velocity_std: float = 0.0
    duration: float = 0.0
    n_notes: int = 0
    complexity_score: float = 0.0  # composite complexity


@dataclass
class PairedSample:
    """A single paired sample: MIDI features + bio-vector + conditioning."""
    midi_path: str
    midi_features: MIDIFeatures
    fasta_header: str
    fasta_source: str
    bio_vector: np.ndarray
    conditioning_vector: np.ndarray
    musical_params: dict


class PairedDatasetCreator:
    """
    Creates paired MIDI-bio-vector dataset.

    Uses a complexity-matching strategy:
    - Extract musical features from all MIDI files
    - Extract bio-vectors from all FASTA sequences
    - Sort both by complexity and match by rank
    """

    def __init__(self,
                 midi_dir: str,
                 fasta_path: str,
                 output_dir: str,
                 config_path: str = None,
                 seed: int = 42,
                 min_midi_duration: float = 30.0,
                 max_midi_duration: float = 300.0):
        """
        Initialize paired dataset creator.

        Args:
            midi_dir: Directory containing MIDI files (e.g., MAESTRO)
            fasta_path: Path to FASTA file or directory
            output_dir: Directory to save paired data
            config_path: Path to pipeline config JSON
            seed: Random seed
            min_midi_duration: Minimum MIDI duration in seconds
            max_midi_duration: Maximum MIDI duration in seconds
        """
        self.midi_dir = Path(midi_dir)
        self.fasta_path = Path(fasta_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.min_midi_duration = min_midi_duration
        self.max_midi_duration = max_midi_duration

        # Load config if available
        self.config = None
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)

        # Initialize components
        extractor_config = self.config['extraction'] if self.config else {}
        self.extractor = BioVectorExtractor(
            kmer_sizes=extractor_config.get('kmer_sizes', [1, 2, 3]),
            window_size=extractor_config.get('window_size', 1000),
            stride=extractor_config.get('stride', 500),
            min_sequence_length=extractor_config.get('min_sequence_length', 100)
        )

        sonification_config = self.config['sonification'] if self.config else {}
        self.mapper = SonificationMapper(
            tempo_range=tuple(sonification_config.get('tempo_range', [60, 180])),
            pitch_range=tuple(sonification_config.get('pitch_range', [36, 96])),
            key_mapping=sonification_config.get('key_mapping', 'cycle_of_fifths'),
            chord_complexity_levels=sonification_config.get('chord_complexity_levels', 5)
        )

        self.preprocessor = MIDIPreprocessor(
            min_duration=min_midi_duration,
            max_duration=max_midi_duration
        )

    def extract_midi_features(self, midi_path: str) -> Optional[MIDIFeatures]:
        """Extract statistical features from a MIDI file."""
        try:
            midi = mido.MidiFile(midi_path)
        except Exception:
            return None

        features = MIDIFeatures(filepath=midi_path)

        # Merge all tracks
        merged = mido.merge_tracks(midi.tracks)

        pitches = []
        velocities = []
        note_times = []
        current_time = 0
        tempo_bpm = 120.0

        for msg in merged:
            current_time += msg.time

            if msg.type == 'set_tempo':
                tempo_bpm = mido.tempo2bpm(msg.tempo)

            if msg.type == 'note_on' and msg.velocity > 0:
                if self.preprocessor.pitch_range[0] <= msg.note <= self.preprocessor.pitch_range[1]:
                    pitches.append(msg.note)
                    velocities.append(msg.velocity)
                    note_times.append(current_time)

        if len(pitches) == 0:
            return None

        # Compute duration in seconds
        if midi.ticks_per_beat > 0:
            seconds_per_tick = 60.0 / (tempo_bpm * midi.ticks_per_beat)
            features.duration = current_time * seconds_per_tick
        else:
            features.duration = 0

        # Check duration constraints
        if features.duration < self.min_midi_duration or features.duration > self.max_midi_duration:
            return None

        # Basic statistics
        features.mean_pitch = float(np.mean(pitches))
        features.std_pitch = float(np.std(pitches))
        features.pitch_range_span = float(max(pitches) - min(pitches))
        features.velocity_mean = float(np.mean(velocities))
        features.velocity_std = float(np.std(velocities))
        features.n_notes = len(pitches)
        features.tempo = tempo_bpm

        # Note density (notes per second)
        features.note_density = len(pitches) / max(features.duration, 1.0)

        # ============ Shannon Entropy-based Complexity ============
        # Scientifically grounded complexity metrics (Madsen & Widmer 2015)
        pitch_entropy = self._compute_pitch_entropy(pitches)
        velocity_entropy = self._compute_velocity_entropy(velocities)

        # Composite complexity: weighted Shannon entropy
        # 60% pitch diversity + 40% dynamic diversity
        complexity = (
            0.6 * pitch_entropy +
            0.4 * velocity_entropy
        )
        features.complexity_score = float(np.clip(complexity, 0, 1))

        return features

    def _compute_pitch_entropy(self, pitches: List[int]) -> float:
        """
        Compute normalized Shannon entropy of pitch classes.

        H = -sum(p(pc) * log2(p(pc))) / log2(12)

        This is scientifically grounded in music information retrieval literature.
        """
        pitch_classes = [p % 12 for p in pitches]
        total = len(pitch_classes)
        pc_counts = Counter(pitch_classes)
        probs = [count / total for count in pc_counts.values()]

        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(12)  # 12 pitch classes

        return float(entropy / max_entropy)

    def _compute_velocity_entropy(self, velocities: List[int]) -> float:
        """
        Compute normalized Shannon entropy of velocity distribution.

        Measures dynamic variety: high = many dynamic levels, low = monotonous.
        """
        total = len(velocities)
        vel_counts = Counter(velocities)
        probs = [count / total for count in vel_counts.values()]

        entropy = -sum(p * np.log2(p) for p in probs if p > 0)
        max_entropy = np.log2(max(len(vel_counts), 2))

        return float(entropy / max_entropy) if max_entropy > 0 else 0.0

    def scan_midi_files(self) -> List[MIDIFeatures]:
        """Scan MIDI directory and extract features from all files."""
        print(f"Scanning MIDI files in: {self.midi_dir}")
        midi_files = list(self.midi_dir.rglob("*.mid")) + list(self.midi_dir.rglob("*.midi"))
        midi_files = sorted(set(midi_files))

        print(f"Found {len(midi_files)} MIDI files, extracting features...")
        midi_features = []

        for midi_path in tqdm(midi_files, desc="Extracting MIDI features"):
            features = self.extract_midi_features(str(midi_path))
            if features is not None:
                midi_features.append(features)

        print(f"Valid MIDI files: {len(midi_features)}")
        return midi_features

    def load_and_extract_bio_vectors(self, n_sequences: int) -> List[Tuple[str, str, np.ndarray]]:
        """
        Load FASTA sequences and extract bio-vectors.

        Args:
            n_sequences: Number of sequences to use (matches MIDI count if available)

        Returns:
            List of (header, source_file, bio_vector) tuples
        """
        print(f"Loading FASTA data from: {self.fasta_path}")

        # Load FASTA sequences
        loader = FastaDatasetLoader(
            min_sequence_length=self.extractor.min_sequence_length,
            max_sequences=n_sequences
        )
        if self.fasta_path.is_file():
            sequences = loader.load_from_directory(str(self.fasta_path.parent), recursive=False)
        else:
            sequences = loader.load_from_directory(str(self.fasta_path), recursive=True)

        print(f"Loaded {len(sequences)} FASTA sequences")

        # Extract bio-vectors
        target_dim = 128
        bio_data = []

        for seq in tqdm(sequences, desc="Extracting bio-vectors"):
            if len(bio_data) >= n_sequences:
                break

            try:
                features = self.extractor.extract_features(seq.sequence)
                bio_vector = self.extractor.create_bio_vector(features, target_dim)
                bio_data.append((seq.header, seq.source_file, bio_vector))
            except Exception as e:
                print(f"Warning: Failed to process {seq.header[:50]}: {e}")

        print(f"Extracted {len(bio_data)} bio-vectors")
        return bio_data

    def create_pairs(self,
                     midi_features: List[MIDIFeatures],
                     bio_data: List[Tuple[str, str, np.ndarray]]) -> List[PairedSample]:
        """
        Create paired samples by complexity matching.

        Strategy:
        1. Sort MIDI files by complexity_score
        2. Sort bio-vectors by statistical complexity (variance + entropy)
        3. Match by rank: most complex MIDI ↔ most complex bio-vector

        Args:
            midi_features: List of MIDI feature objects
            bio_data: List of (header, source, bio_vector) tuples

        Returns:
            List of PairedSample objects
        """
        n_pairs = min(len(midi_features), len(bio_data))
        print(f"Creating {n_pairs} paired samples...")

        # Sort MIDI by complexity
        midi_sorted = sorted(midi_features, key=lambda x: x.complexity_score)

        # Compute bio-vector complexity and sort
        bio_complexity = []
        for header, source, bio_vec in bio_data:
            # Complexity = variance + magnitude of non-zero elements
            complexity = float(np.var(bio_vec)) + float(np.mean(np.abs(bio_vec)))
            bio_complexity.append((header, source, bio_vec, complexity))

        bio_sorted = sorted(bio_complexity, key=lambda x: x[3])

        # Match by rank
        pairs = []
        for i in range(n_pairs):
            midi_feat = midi_sorted[i]
            header, source, bio_vec, _ = bio_sorted[i]

            # Create conditioning vector
            musical_params = self.mapper.bio_vector_to_musical_params(bio_vec)
            cond_vec = self.mapper.create_conditioning_vector(musical_params)

            pair = PairedSample(
                midi_path=midi_feat.filepath,
                midi_features=midi_feat,
                fasta_header=header,
                fasta_source=source,
                bio_vector=bio_vec,
                conditioning_vector=cond_vec,
                musical_params={
                    'key': musical_params.key,
                    'tempo': musical_params.tempo,
                    'pitch_range': list(musical_params.pitch_range),
                    'rhythm_complexity': musical_params.rhythm_complexity,
                    'scale_type': musical_params.scale_type,
                    'articulation_density': musical_params.articulation_density,
                }
            )
            pairs.append(pair)

        print(f"Created {len(pairs)} pairs")
        return pairs

    def save_paired_dataset(self, pairs: List[PairedSample]):
        """
        Save paired dataset to files.

        Creates:
        - paired_data.json: metadata for each pair
        - bio_vectors.npy: all bio-vectors (aligned with JSON order)
        - conditioning_vectors.npy: all conditioning vectors
        - pairs_manifest.csv: human-readable manifest
        """
        print(f"Saving paired dataset to: {self.output_dir}")

        # Save bio-vectors
        bio_vectors = np.array([p.bio_vector for p in pairs])
        np.save(self.output_dir / 'paired_bio_vectors.npy', bio_vectors)

        # Save conditioning vectors
        cond_vectors = np.array([p.conditioning_vector for p in pairs])
        np.save(self.output_dir / 'paired_conditioning_vectors.npy', cond_vectors)

        # Save metadata
        metadata = []
        for p in pairs:
            musical_params = {
                'key': str(p.musical_params['key']),
                'tempo': float(p.musical_params['tempo']),
                'pitch_range': [int(x) for x in p.musical_params['pitch_range']],
                'rhythm_complexity': float(p.musical_params['rhythm_complexity']),
                'scale_type': str(p.musical_params['scale_type']),
                'articulation_density': float(p.musical_params['articulation_density']),
            }
            metadata.append({
                'midi_path': p.midi_path,
                'midi_complexity': float(p.midi_features.complexity_score),
                'midi_note_density': float(p.midi_features.note_density),
                'midi_mean_pitch': float(p.midi_features.mean_pitch),
                'midi_tempo': float(p.midi_features.tempo),
                'midi_n_notes': int(p.midi_features.n_notes),
                'fasta_header': p.fasta_header,
                'fasta_source': p.fasta_source,
                'musical_params': musical_params
            })

        with open(self.output_dir / 'paired_data.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save manifest CSV
        with open(self.output_dir / 'pairs_manifest.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'midi_path', 'midi_complexity', 'fasta_header',
                'fasta_source', 'key', 'tempo', 'scale_type', 'rhythm_complexity'
            ])
            for p in pairs:
                writer.writerow([
                    p.midi_path,
                    f"{p.midi_features.complexity_score:.4f}",
                    p.fasta_header[:100],
                    Path(p.fasta_source).name,
                    p.musical_params['key'],
                    f"{p.musical_params['tempo']:.1f}",
                    p.musical_params['scale_type'],
                    f"{p.musical_params['rhythm_complexity']:.4f}"
                ])

        # Save statistics
        stats = {
            'n_pairs': len(pairs),
            'midi_complexity_range': [
                float(min(p.midi_features.complexity_score for p in pairs)),
                float(max(p.midi_features.complexity_score for p in pairs))
            ],
            'midi_note_density_range': [
                float(min(p.midi_features.note_density for p in pairs)),
                float(max(p.midi_features.note_density for p in pairs))
            ],
            'key_distribution': self._compute_key_distribution(pairs),
            'scale_type_distribution': self._compute_scale_distribution(pairs),
            'bio_vector_dim': bio_vectors.shape[1],
            'conditioning_dim': cond_vectors.shape[1],
        }

        with open(self.output_dir / 'paired_stats.json', 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Saved paired dataset:")
        print(f"  Pairs: {len(pairs)}")
        print(f"  Bio-vector dim: {bio_vectors.shape[1]}")
        print(f"  Conditioning dim: {cond_vectors.shape[1]}")
        print(f"  Key distribution: {stats['key_distribution']}")

    def _compute_key_distribution(self, pairs: List[PairedSample]) -> Dict[str, int]:
        from collections import Counter
        keys = [p.musical_params['key'] for p in pairs]
        return dict(Counter(keys))

    def _compute_scale_distribution(self, pairs: List[PairedSample]) -> Dict[str, int]:
        from collections import Counter
        scales = [p.musical_params['scale_type'] for p in pairs]
        return dict(Counter(scales))

    def run(self):
        """Execute full pairing pipeline."""
        set_seed(self.seed)

        print("=" * 60)
        print("PAIRED DATASET CREATOR")
        print("=" * 60)

        # Step 1: Extract MIDI features
        midi_features = self.scan_midi_files()
        if len(midi_features) == 0:
            raise ValueError("No valid MIDI files found!")

        # Step 2: Load FASTA and extract bio-vectors
        bio_data = self.load_and_extract_bio_vectors(n_sequences=len(midi_features))
        if len(bio_data) == 0:
            raise ValueError("No valid FASTA sequences found!")

        # Step 3: Create pairs
        pairs = self.create_pairs(midi_features, bio_data)

        # Step 4: Save
        self.save_paired_dataset(pairs)

        print("\nPaired dataset creation complete!")
        return pairs


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Create paired MIDI-bio-vector dataset'
    )
    parser.add_argument(
        '--midi-dir',
        type=str,
        default='data/midi/maestro-v3.0.0',
        help='Directory containing MIDI files'
    )
    parser.add_argument(
        '--fasta-path',
        type=str,
        default='data/fasta/training/Homo_sapiens.GRCh38.cds.all.fa',
        help='Path to FASTA file or directory'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='results/paired_data',
        help='Output directory for paired data'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/pipeline_config.json',
        help='Path to pipeline config'
    )
    parser.add_argument(
        '--min-duration',
        type=float,
        default=30.0,
        help='Minimum MIDI duration in seconds'
    )
    parser.add_argument(
        '--max-duration',
        type=float,
        default=300.0,
        help='Maximum MIDI duration in seconds'
    )

    args = parser.parse_args()

    creator = PairedDatasetCreator(
        midi_dir=args.midi_dir,
        fasta_path=args.fasta_path,
        output_dir=args.output_dir,
        config_path=args.config,
        min_midi_duration=args.min_duration,
        max_midi_duration=args.max_duration
    )

    creator.run()


if __name__ == '__main__':
    main()
