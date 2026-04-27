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
from .structured_generate import generate_structured_music_from_fasta
from .structured_model import BioConditionedSequenceModel
from .structured_music import (
    HarmonyBar,
    HarmonyTokenizer,
    MelodyEvent,
    MelodyTokenizer,
    StructuredMusicSegment,
    load_structured_music_corpus,
    render_harmony_and_melody_to_score,
)
from .structured_pairing import StructuredPairedSample, build_structured_paired_dataset
from .structured_train import train_structured_pipeline
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
    "HarmonyBar",
    "HarmonyTokenizer",
    "MelodyEvent",
    "MelodyTokenizer",
    "StructuredMusicSegment",
    "load_structured_music_corpus",
    "render_harmony_and_melody_to_score",
    "StructuredPairedSample",
    "build_structured_paired_dataset",
    "BioConditionedSequenceModel",
    "train_pipeline",
    "train_structured_pipeline",
    "generate_structured_music_from_fasta",
]
