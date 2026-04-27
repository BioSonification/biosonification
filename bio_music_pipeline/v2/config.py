"""Configuration for the next-generation biosonification pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BioEncoderConfig:
    min_sequence_length: int = 90
    max_sequence_length: int = 12000
    fragment_length: int = 1800
    fragment_stride: int = 900
    max_fragments_per_record: int = 24
    kmer_sizes: List[int] = field(default_factory=lambda: [1, 2, 3])
    max_kmer_features: int = 84
    use_vienna_rna: bool = True
    use_protein_features: bool = True
    translate_longest_orf: bool = True
    use_esm_embedding: bool = False
    esm_model_name: str = "facebook/esm2_t6_8M_UR50D"
    esm_feature_dim: int = 64
    esm_max_length: int = 512
    esm_device: str = "auto"
    embedding_dim: int = 256


@dataclass
class MusicDataConfig:
    midi_dirs: List[str] = field(default_factory=lambda: ["data/midi/polyphonic_music21"])
    use_music21_corpus_fallback: bool = True
    music21_composers: List[str] = field(default_factory=lambda: ["bach"])
    max_music21_files: int = 96
    bars_per_segment: int = 8
    segment_hop_bars: int = 4
    steps_per_beat: int = 4
    steps_per_bar: int = 16
    min_notes_per_segment: int = 12
    min_polyphony_ratio: float = 0.1
    pitch_range_min: int = 24
    pitch_range_max: int = 96
    velocity_bins: int = 8
    duration_bins: int = 32
    descriptor_bins: int = 8
    max_events: int = 768
    prefer_named_melody_parts: bool = True
    melody_octave_min: int = 3
    melody_octave_max: int = 7
    chord_octave: int = 3
    chord_hold_token_enabled: bool = True


@dataclass
class PairingConfig:
    top_k: int = 3
    temperature: float = 0.25
    descriptor_weight_harmony: float = 1.2
    descriptor_weight_polyphony: float = 1.1
    descriptor_weight_density: float = 1.0
    descriptor_weight_register: float = 0.9
    descriptor_weight_tempo: float = 0.8
    descriptor_weight_mode: float = 0.6


@dataclass
class TrainingConfig:
    seed: int = 42
    device: str = "auto"
    num_epochs: int = 20
    harmony_num_epochs: int = 16
    melody_num_epochs: int = 20
    batch_size: int = 4
    grad_accum_steps: int = 4
    learning_rate: float = 3e-4
    harmony_learning_rate: float = 3e-4
    melody_learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    val_fraction: float = 0.15
    test_fraction: float = 0.1
    patience: int = 5
    mixed_precision: bool = True
    num_workers: int = 0
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.15
    max_seq_len: int = 896
    harmony_max_seq_len: int = 128
    melody_max_seq_len: int = 640


@dataclass
class GenerationConfig:
    num_bars: int = 8
    harmony_max_new_tokens: int = 64
    melody_max_new_tokens: int = 256
    harmony_temperature: float = 0.9
    melody_temperature: float = 0.92
    harmony_top_k: int = 12
    melody_top_k: int = 24
    harmony_top_p: float = 0.9
    melody_top_p: float = 0.92
    melody_min_new_tokens: int = 48


@dataclass
class V2PipelineConfig:
    output_dir: str = "results/v2_default"
    fasta_path: str = "data/fasta/quick_sample.fa"
    bio: BioEncoderConfig = field(default_factory=BioEncoderConfig)
    music: MusicDataConfig = field(default_factory=MusicDataConfig)
    pairing: PairingConfig = field(default_factory=PairingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


def _merge_dataclass(instance: Any, payload: Dict[str, Any]) -> Any:
    values = {}
    for field_info in instance.__dataclass_fields__.values():
        name = field_info.name
        current_value = getattr(instance, name)
        if name not in payload:
            values[name] = current_value
            continue
        incoming = payload[name]
        if hasattr(current_value, "__dataclass_fields__") and isinstance(incoming, dict):
            values[name] = _merge_dataclass(current_value, incoming)
        else:
            values[name] = incoming
    return type(instance)(**values)


def load_v2_config(config_path: Optional[str] = None) -> V2PipelineConfig:
    """Load the v2 config with defaults."""

    config = V2PipelineConfig()
    if config_path is None:
        return config
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return _merge_dataclass(config, payload)
