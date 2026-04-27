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
    v2_config_from_dict,
)
from .evaluate import (
    StructuredEvaluationConfig,
    compute_structured_midi_metrics,
    run_structured_evaluation,
)
from .dataset_report import DatasetReportConfig, build_dataset_report
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
    "v2_config_from_dict",
    "StructuredEvaluationConfig",
    "compute_structured_midi_metrics",
    "run_structured_evaluation",
    "DatasetReportConfig",
    "build_dataset_report",
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
    "train_structured_pipeline",
    "generate_structured_music_from_fasta",
]
