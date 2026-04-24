"""Next-generation biosonification pipeline."""

from .bio import BioEncodingResult, BiologicalSequenceEncoder
from .config import (
    BioEncoderConfig,
    GenerationConfig,
    MusicDataConfig,
    PairingConfig,
    TrainingConfig,
    V2PipelineConfig,
    load_v2_config,
)
from .dataset import (
    BioMusicPairDataset,
    MusicSegment,
    PolyphonicMusicTokenizer,
    load_music_corpus,
)
from .generate import generate_music_from_fasta
from .model import ControlConditionedTransformer
from .pairing import PairedSample, build_paired_dataset
from .train import train_pipeline

__all__ = [
    "BioEncodingResult",
    "BiologicalSequenceEncoder",
    "BioEncoderConfig",
    "GenerationConfig",
    "MusicDataConfig",
    "PairingConfig",
    "TrainingConfig",
    "V2PipelineConfig",
    "load_v2_config",
    "BioMusicPairDataset",
    "MusicSegment",
    "PolyphonicMusicTokenizer",
    "load_music_corpus",
    "generate_music_from_fasta",
    "ControlConditionedTransformer",
    "PairedSample",
    "build_paired_dataset",
    "train_pipeline",
]
