"""
Bio-Music Generator - Web Interface Backend

Wraps the bio_music_pipeline to generate MIDI from FASTA sequences
without retraining the model.
"""

import re
import json
import uuid
import os
import torch
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from bio_music_pipeline import set_seed
from bio_music_pipeline.extractors import BioVectorExtractor
from bio_music_pipeline.sonification import SonificationMapper
from bio_music_pipeline.models import create_bio_music_model
from bio_music_pipeline.data import MIDIPreprocessor
from bio_music_pipeline.utils import tokens_to_midi


class FASTAValidationError(Exception):
    """Custom exception for FASTA validation errors."""
    pass


class BioMusicGenerator:
    """
    Generates MIDI music from biological FASTA sequences.
    
    Loads the trained model once and reuses it for multiple generations.
    """

    def __init__(self, config_path: Optional[str] = None, model_path: Optional[str] = None, seed: int = 42):
        """
        Initialize the generator.
        
        Args:
            config_path: Path to pipeline config JSON
            model_path: Path to trained model checkpoint
            seed: Random seed for reproducibility
        """
        self.seed = seed
        set_seed(seed)

        # Resolve paths
        self.config_path = Path(config_path) if config_path else self._resolve_default_config_path()
        self.model_path = Path(model_path) if model_path else self._resolve_default_model_path()

        # State
        self.config = None
        self.model = None
        self.device = None
        self.extractor = None
        self.mapper = None
        self.preprocessor = None
        self._initialized = False
        self._error = None

        # Validate paths
        if not self.config_path.exists():
            self._error = f"Config file not found: {self.config_path}"
        if not self.model_path.exists():
            self._error = (
                f"Trained model not found: {self.model_path}. "
                "Set BIOSONIFICATION_MODEL_PATH or run the full pipeline first."
            )

    @staticmethod
    def _resolve_default_config_path() -> Path:
        """
        Pick the most relevant config for web inference.

        Priority:
        1. full paired config (current main path)
        2. legacy pipeline config
        """
        candidates = [
            PROJECT_ROOT / "configs" / "pipeline_full_paired.json",
            PROJECT_ROOT / "configs" / "pipeline_config.json",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    @staticmethod
    def _resolve_default_model_path() -> Path:
        """
        Pick default trained model checkpoint.

        Priority:
        1. Explicit BIOSONIFICATION_MODEL_PATH env var
        2. Known canonical paths
        3. Newest results/*/models/best_model.pt fallback
        """
        env_model = (Path(os.environ["BIOSONIFICATION_MODEL_PATH"]).expanduser()
                     if "BIOSONIFICATION_MODEL_PATH" in os.environ else None)
        if env_model is not None:
            return env_model

        known_candidates = [
            PROJECT_ROOT / "results" / "full_paired_run" / "models" / "best_model.pt",
            PROJECT_ROOT / "results" / "models" / "best_model.pt",
        ]
        existing_known = [p for p in known_candidates if p.exists()]
        if existing_known:
            return max(existing_known, key=lambda p: p.stat().st_mtime)

        fallback_candidates = list((PROJECT_ROOT / "results").glob("*/models/best_model.pt"))
        if fallback_candidates:
            return max(fallback_candidates, key=lambda p: p.stat().st_mtime)

        return known_candidates[0]

    def initialize(self) -> bool:
        """
        Load model and initialize components.
        
        Returns:
            True if initialization successful
        """
        if self._error:
            return False

        if self._initialized:
            return True

        try:
            # Load config
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)

            # Load model
            try:
                checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            except TypeError:
                # Backward compatibility with older torch versions.
                checkpoint = torch.load(self.model_path, map_location='cpu')
            model_config = checkpoint['config']

            # Initialize extractor
            extraction_config = self.config['extraction']
            self.extractor = BioVectorExtractor(
                kmer_sizes=extraction_config['kmer_sizes'],
                window_size=extraction_config['window_size'],
                stride=extraction_config['stride'],
                min_sequence_length=extraction_config['min_sequence_length']
            )

            # Initialize sonification mapper
            sonification_config = self.config['sonification']
            self.mapper = SonificationMapper(
                tempo_range=tuple(sonification_config['tempo_range']),
                pitch_range=tuple(sonification_config['pitch_range']),
                key_mapping=sonification_config['key_mapping'],
                chord_complexity_levels=sonification_config['chord_complexity_levels']
            )

            # Initialize deterministic tokenizer vocabulary without rescanning all MIDI.
            self.preprocessor = MIDIPreprocessor(
                max_seq_len=int(model_config.get('max_seq_len', self.config['model']['max_seq_len'])),
                min_duration=float(self.config['data']['min_midi_duration']),
                max_duration=float(self.config['data']['max_midi_duration'])
            )
            expected_vocab_size = self.preprocessor.vocab_size
            checkpoint_vocab_size = int(model_config.get('vocab_size', expected_vocab_size))
            if checkpoint_vocab_size != expected_vocab_size:
                raise RuntimeError(
                    f"Checkpoint vocab_size={checkpoint_vocab_size} does not match "
                    f"expected tokenizer vocab_size={expected_vocab_size}."
                )

            # Ensure token metadata is present in config (backward compatibility).
            model_config['vocab_size'] = checkpoint_vocab_size
            model_config.setdefault('bos_token_id', self.preprocessor.bos_token_id)
            model_config.setdefault('eos_token_id', self.preprocessor.eos_token_id)
            model_config.setdefault('pad_token_id', self.preprocessor.pad_token_id)

            self.device = torch.device('cpu')
            self.model = create_bio_music_model(model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            self._initialized = True
            return True

        except Exception as e:
            self._error = f"Initialization error: {str(e)}"
            import traceback
            traceback.print_exc()
            return False

    @staticmethod
    def validate_fasta(fasta_text: str) -> Tuple[str, str]:
        """
        Validate FASTA text and extract sequence.
        
        Args:
            fasta_text: FASTA format text
            
        Returns:
            Tuple of (header, sequence)
            
        Raises:
            FASTAValidationError: If validation fails
        """
        fasta_text = fasta_text.strip()
        
        if not fasta_text:
            raise FASTAValidationError("FASTA input is empty")

        # Check if it looks like FASTA format
        if not fasta_text.startswith('>'):
            # Try to treat it as raw sequence
            sequence = re.sub(r'[^ACGTacgt]', '', fasta_text).upper()
            if len(sequence) < 100:
                raise FASTAValidationError(
                    f"Sequence too short: {len(sequence)} nucleotides (minimum 100). "
                    "Please provide a longer DNA sequence."
                )
            return "User Sequence", sequence
        
        # Parse FASTA format with support for multi-record input.
        records = []
        current_header = None
        current_sequence_lines = []
        for raw_line in fasta_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header is not None:
                    records.append((current_header, ''.join(current_sequence_lines)))
                current_header = line[1:].strip() or f"Sequence_{len(records) + 1}"
                current_sequence_lines = []
            else:
                if current_header is None:
                    raise FASTAValidationError(
                        "Invalid FASTA format: sequence content must come after a header line starting with '>'."
                    )
                current_sequence_lines.append(line)

        if current_header is not None:
            records.append((current_header, ''.join(current_sequence_lines)))

        if not records:
            raise FASTAValidationError("Invalid FASTA input: no valid FASTA records found.")

        # Use the first non-empty cleaned record; do not concatenate multiple records.
        cleaned_records = []
        for header, raw_sequence in records:
            cleaned = re.sub(r'[^ACGTacgt]', '', raw_sequence).upper()
            if cleaned:
                cleaned_records.append((header, cleaned))

        if not cleaned_records:
            raise FASTAValidationError("No valid DNA bases found in FASTA records (expected A/C/G/T).")

        header, sequence = cleaned_records[0]
        if len(cleaned_records) > 1:
            header = f"{header} (record 1/{len(cleaned_records)})"
        
        if len(sequence) < 100:
            raise FASTAValidationError(
                f"Sequence too short: {len(sequence)} nucleotides (minimum 100). "
                "Please provide a longer DNA sequence."
            )
        
        if len(sequence) > 50000:
            raise FASTAValidationError(
                f"Sequence too long: {len(sequence)} nucleotides (maximum 50000). "
                "Very long sequences may take a while to process."
            )
        
        return header, sequence

    def generate(self, fasta_text: str, output_dir: Optional[str] = None) -> Dict:
        """
        Generate MIDI music from FASTA sequence.
        
        Args:
            fasta_text: FASTA format text or raw DNA sequence
            output_dir: Directory to save output files
            
        Returns:
            Dictionary with generation results:
            - session_id: Unique identifier
            - midi_path: Path to MIDI file
            - audio_path: Path to WAV audio file (if conversion successful)
            - musical_params: Dict of musical parameters
            - header: FASTA header
            - sequence_length: Length of input sequence
        """
        if not self._initialized:
            if not self.initialize():
                raise RuntimeError(f"Generator not initialized: {self._error}")

        # Validate FASTA
        header, sequence = self.validate_fasta(fasta_text)

        # Create session ID
        session_id = str(uuid.uuid4())[:8]

        # Setup output directory
        if output_dir is None:
            output_dir = PROJECT_ROOT / "web" / "output"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        midi_dir = output_dir / "midi"
        midi_dir.mkdir(parents=True, exist_ok=True)

        # Extract bio-vector
        features = self.extractor.extract_features(sequence)
        target_dim = self.config['model']['bio_vector_dim']
        bio_vector = self.extractor.create_bio_vector(features, target_dim)

        # Apply sonification rules
        musical_params = self.mapper.bio_vector_to_musical_params(bio_vector)

        # Generate tokens
        bio_tensor = torch.tensor(bio_vector, dtype=torch.float32).unsqueeze(0).to(self.device)
        max_len = self.config['model']['max_seq_len']

        with torch.no_grad():
            generated_tokens = self.model.generate(
                bio_tensor,
                max_len=max_len,
                temperature=1.0,
                use_gumbel=False
            )

        generated_tokens = generated_tokens.squeeze(0).numpy()

        # Convert to MIDI
        vocab = self.preprocessor.token_to_idx
        midi_path = midi_dir / f"{session_id}.mid"
        
        tokens_list = generated_tokens.tolist()
        success = tokens_to_midi(
            tokens_list,
            vocab,
            str(midi_path),
            tempo=int(musical_params.tempo)
        )

        if not success:
            raise RuntimeError("Failed to convert tokens to MIDI file")

        # Prepare result
        result = {
            'session_id': session_id,
            'midi_path': str(midi_path),
            'midi_filename': f"{session_id}.mid",
            'header': header,
            'sequence_length': len(sequence),
            'num_tokens_generated': len(generated_tokens),
            'musical_params': {
                'key': musical_params.key,
                'tempo': float(musical_params.tempo),
                'pitch_range': list(musical_params.pitch_range),
                'rhythm_complexity': float(musical_params.rhythm_complexity),
                'scale_type': musical_params.scale_type,
                'articulation_density': float(musical_params.articulation_density),
                'dynamic_range': list(musical_params.dynamic_range),
            }
        }

        return result

    def is_ready(self) -> bool:
        """Check if generator is ready to use."""
        return self._initialized and self._error is None

    def get_error(self) -> Optional[str]:
        """Get initialization error message."""
        return self._error


# Module-level singleton for reuse
_generator = None


def get_generator() -> BioMusicGenerator:
    """Get or create the generator singleton."""
    global _generator
    if _generator is None:
        _generator = BioMusicGenerator()
    return _generator
