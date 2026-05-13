"""Biologically informed sequence encoding for the v2 pipeline."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from .config import BioEncoderConfig

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqUtils.ProtParam import ProteinAnalysis
except ImportError:  # pragma: no cover - exercised in runtime bootstrap
    SeqIO = None
    Seq = None
    ProteinAnalysis = None

try:
    import RNA  # type: ignore
except ImportError:  # pragma: no cover - ViennaRNA is optional
    RNA = None


NUCLEOTIDES = ("A", "C", "G", "T")
RNA_BASES = ("A", "C", "G", "U")
AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
STOP_CODONS = {"TAA", "TAG", "TGA"}


@dataclass
class BioEncodingResult:
    sequence_id: str
    sequence_type: str
    cleaned_sequence: str
    vector: np.ndarray
    control_profile: np.ndarray
    tonic_pc_hint: int
    feature_names: List[str]
    feature_map: Dict[str, float]
    translated_protein: str
    predicted_structure: str


def _safe_entropy(values: Sequence[float]) -> float:
    total = float(sum(values))
    if total <= 0:
        return 0.0
    entropy = 0.0
    for value in values:
        if value <= 0:
            continue
        probability = value / total
        entropy -= probability * math.log2(probability)
    return entropy


def _normalized(value: float, lower: float, upper: float) -> float:
    if upper <= lower:
        return 0.0
    return float(np.clip((value - lower) / (upper - lower), 0.0, 1.0))


class BiologicalSequenceEncoder:
    """Extracts biologically informed embeddings from FASTA sequences."""

    def __init__(self, config: Optional[BioEncoderConfig] = None):
        self.config = config or BioEncoderConfig()
        if SeqIO is None or ProteinAnalysis is None or Seq is None:
            raise ImportError(
                "Biopython is required for the v2 sequence encoder. " "Install it before running the pipeline."
            )
        self._esm_tokenizer = None
        self._esm_model = None
        self._esm_device = None
        self._esm_cache: Dict[str, np.ndarray] = {}

    def _parse_fasta(self, fasta_path: str) -> List[tuple[str, str]]:
        records: List[tuple[str, str]] = []
        path = Path(fasta_path)

        # Handle both single file and directory
        if path.is_dir():
            # Directory with multiple FASTA files
            fasta_files = sorted(path.glob("*.fna")) + sorted(path.glob("*.fa")) + sorted(path.glob("*.fasta"))
            for fasta_file in fasta_files:
                for record in SeqIO.parse(str(fasta_file), "fasta"):
                    sequence = str(record.seq).strip()
                    if not sequence:
                        continue
                    records.append((record.id, sequence))
        else:
            # Single FASTA file
            for record in SeqIO.parse(fasta_path, "fasta"):
                sequence = str(record.seq).strip()
                if not sequence:
                    continue
                records.append((record.id, sequence))
        return records

    def encode_fasta(self, fasta_path: str) -> List[BioEncodingResult]:
        path = Path(fasta_path)
        if not path.exists():
            raise FileNotFoundError(f"FASTA file not found: {path}")

        # Parse all records first to show progress
        records = self._parse_fasta(str(path))
        print(f"Loaded {len(records)} FASTA records, extracting fragments...")

        results: List[BioEncodingResult] = []
        skipped_count = 0

        # Process with progress bar
        for record_id, sequence in tqdm(records, desc="Processing sequences", unit="seq"):
            fragments = list(self._fragment_sequence(sequence))
            for fragment_index, fragment in enumerate(fragments):
                if len(fragment) < self.config.min_sequence_length:
                    continue
                fragment_id = f"{record_id}::frag{fragment_index:03d}"
                try:
                    results.append(self.encode_sequence(fragment, sequence_id=fragment_id))
                except ValueError:
                    # Skip fragments that are too short after cleaning
                    skipped_count += 1
                    continue

        if not results:
            raise ValueError(
                f"No valid sequences found in {path}. "
                f"Expected at least one record with length >= {self.config.min_sequence_length}."
            )

        print(f"Extracted {len(results)} biological fragments (skipped {skipped_count} short fragments)")
        return results

    def _fragment_sequence(self, sequence: str) -> List[str]:
        sequence = re.sub(r"\s+", "", sequence).upper()
        fragment_length = min(self.config.fragment_length, self.config.max_sequence_length)
        if fragment_length <= 0 or len(sequence) <= fragment_length:
            return [sequence[: self.config.max_sequence_length]]

        fragments: List[str] = []
        stride = max(self.config.min_sequence_length, self.config.fragment_stride)
        for start in range(0, len(sequence), stride):
            fragment = sequence[start : start + fragment_length]
            if len(fragment) < self.config.min_sequence_length:
                break
            fragments.append(fragment)
            if len(fragments) >= self.config.max_fragments_per_record:
                break
        if not fragments:
            fragments.append(sequence[: self.config.max_sequence_length])
        return fragments

    def infer_sequence_type(self, sequence: str) -> str:
        clean = re.sub(r"\s+", "", sequence).upper()
        dna_set = set("ACGTN")
        rna_set = set("ACGUN")
        aa_set = set("ABCDEFGHIKLMNPQRSTVWXYZ*")
        chars = set(clean)
        if chars and chars <= dna_set:
            return "dna"
        if chars and chars <= rna_set:
            return "rna"
        if chars and chars <= aa_set:
            return "protein"
        nucleotide_ratio = sum(char in dna_set or char in rna_set for char in clean) / max(len(clean), 1)
        return "dna" if nucleotide_ratio >= 0.85 else "protein"

    def _clean_sequence(self, sequence: str, sequence_type: str) -> str:
        sequence = re.sub(r"\s+", "", sequence).upper()
        if sequence_type == "protein":
            return "".join(char for char in sequence if char in AMINO_ACIDS)
        valid = RNA_BASES if sequence_type == "rna" else NUCLEOTIDES
        cleaned = "".join(char for char in sequence if char in valid)
        if sequence_type == "rna":
            return cleaned.replace("T", "U")
        return cleaned.replace("U", "T")

    def _kmer_distribution(self, sequence: str, k: int, alphabet: Sequence[str]) -> np.ndarray:
        if len(sequence) < k:
            return np.zeros(len(alphabet) ** k, dtype=np.float32)
        counts = Counter(sequence[index : index + k] for index in range(len(sequence) - k + 1))
        kmers: List[str] = [""]
        for _ in range(k):
            kmers = [prefix + letter for prefix in kmers for letter in alphabet]
        total = sum(counts[kmer] for kmer in kmers)
        if total == 0:
            return np.zeros(len(kmers), dtype=np.float32)
        return np.array([counts[kmer] / total for kmer in kmers], dtype=np.float32)

    def _longest_orf(self, dna_sequence: str) -> str:
        if not self.config.translate_longest_orf:
            return ""
        best = ""
        dna_sequence = dna_sequence.replace("U", "T")
        for frame in range(3):
            candidate = ""
            in_orf = False
            for index in range(frame, len(dna_sequence) - 2, 3):
                codon = dna_sequence[index : index + 3]
                if len(codon) < 3:
                    continue
                if not in_orf and codon == "ATG":
                    in_orf = True
                    candidate = "ATG"
                    continue
                if not in_orf:
                    continue
                if codon in STOP_CODONS:
                    if len(candidate) > len(best):
                        best = candidate
                    candidate = ""
                    in_orf = False
                    continue
                candidate += codon
            if in_orf and len(candidate) > len(best):
                best = candidate
        if len(best) < 6:
            return ""
        protein = str(Seq(best).translate(to_stop=True))
        return "".join(char for char in protein if char in AMINO_ACIDS)

    def _protein_feature_block(self, protein_sequence: str) -> Dict[str, float]:
        if not protein_sequence:
            return {
                "protein_length_norm": 0.0,
                "protein_orf_coverage": 0.0,
                "protein_aromaticity": 0.0,
                "protein_instability": 0.5,
                "protein_isoelectric_point": 0.5,
                "protein_gravy": 0.5,
                "protein_molecular_weight": 0.0,
                "protein_flexibility_mean": 0.0,
                "protein_flexibility_std": 0.0,
                "protein_helix": 0.0,
                "protein_turn": 0.0,
                "protein_sheet": 0.0,
                "protein_secondary_diversity": 0.0,
                "protein_charge_density": 0.0,
                **{f"aa_{acid}": 0.0 for acid in AMINO_ACIDS},
            }

        analysis = ProteinAnalysis(protein_sequence)
        aa_percent_source = getattr(analysis, "amino_acids_percent", None)
        if callable(aa_percent_source):
            aa_percent = aa_percent_source()
        elif aa_percent_source is not None:
            aa_percent = aa_percent_source
        else:
            aa_percent = analysis.get_amino_acids_percent()
        flexibility = analysis.flexibility()
        helix, turn, sheet = analysis.secondary_structure_fraction()
        acidic = protein_sequence.count("D") + protein_sequence.count("E")
        basic = protein_sequence.count("K") + protein_sequence.count("R") + protein_sequence.count("H")
        feature_block = {
            "protein_length_norm": _normalized(len(protein_sequence), 20, 1200),
            "protein_orf_coverage": 1.0,
            "protein_aromaticity": float(analysis.aromaticity()),
            "protein_instability": _normalized(float(analysis.instability_index()), 10.0, 80.0),
            "protein_isoelectric_point": _normalized(float(analysis.isoelectric_point()), 3.0, 12.0),
            "protein_gravy": _normalized(float(analysis.gravy()), -2.5, 2.5),
            "protein_molecular_weight": _normalized(float(analysis.molecular_weight()), 2000.0, 180000.0),
            "protein_flexibility_mean": float(np.mean(flexibility)) if flexibility else 0.0,
            "protein_flexibility_std": float(np.std(flexibility)) if flexibility else 0.0,
            "protein_helix": float(helix),
            "protein_turn": float(turn),
            "protein_sheet": float(sheet),
            "protein_secondary_diversity": _safe_entropy([helix, turn, sheet]) / np.log2(3),
            "protein_charge_density": (basic - acidic) / max(len(protein_sequence), 1),
        }
        for acid in AMINO_ACIDS:
            feature_block[f"aa_{acid}"] = float(aa_percent.get(acid, 0.0))
        return feature_block

    def _resolve_esm_device(self) -> str:
        if self.config.esm_device != "auto":
            return self.config.esm_device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load_esm(self):
        if self._esm_model is not None and self._esm_tokenizer is not None:
            return
        try:
            from transformers import AutoTokenizer, EsmModel
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "transformers is required when bio.use_esm_embedding=true. "
                "Install transformers before enabling ESM embeddings."
            ) from exc

        self._esm_device = self._resolve_esm_device()
        self._esm_tokenizer = AutoTokenizer.from_pretrained(self.config.esm_model_name)
        self._esm_model = EsmModel.from_pretrained(self.config.esm_model_name)
        self._esm_model.eval()
        self._esm_model.to(self._esm_device)

    def _reduce_embedding(self, embedding: np.ndarray, target_dim: int) -> np.ndarray:
        if embedding.size == 0 or target_dim <= 0:
            return np.zeros(target_dim, dtype=np.float32)
        if embedding.size == target_dim:
            return embedding.astype(np.float32)
        if embedding.size < target_dim:
            padded = np.zeros(target_dim, dtype=np.float32)
            padded[: embedding.size] = embedding.astype(np.float32)
            return padded
        chunks = np.array_split(embedding.astype(np.float32), target_dim)
        return np.array([float(np.mean(chunk)) for chunk in chunks], dtype=np.float32)

    def _esm_embedding_block(self, protein_sequence: str) -> Dict[str, float]:
        if not self.config.use_esm_embedding or not protein_sequence:
            return {f"esm_{index:03d}": 0.0 for index in range(self.config.esm_feature_dim)}
        if protein_sequence in self._esm_cache:
            vector = self._esm_cache[protein_sequence]
            return {f"esm_{index:03d}": float(value) for index, value in enumerate(vector)}

        self._load_esm()
        import torch

        clipped = protein_sequence[: self.config.esm_max_length]
        inputs = self._esm_tokenizer(
            clipped,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.esm_max_length,
        )
        inputs = {key: value.to(self._esm_device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = self._esm_model(**inputs)
            hidden = outputs.last_hidden_state[0]
            if hidden.size(0) > 2:
                pooled = hidden[1:-1].mean(dim=0).detach().cpu().numpy()
            else:
                pooled = hidden.mean(dim=0).detach().cpu().numpy()
        reduced = self._reduce_embedding(pooled, self.config.esm_feature_dim)
        self._esm_cache[protein_sequence] = reduced
        return {f"esm_{index:03d}": float(value) for index, value in enumerate(reduced)}

    def _rna_structure_block(self, rna_sequence: str) -> tuple[Dict[str, float], str]:
        if not rna_sequence or RNA is None or not self.config.use_vienna_rna:
            return {
                "rna_mfe": 0.0,
                "rna_paired_fraction": 0.0,
                "rna_loop_fraction": 0.0,
                "rna_transition_rate": 0.0,
                "rna_structure_entropy": 0.0,
            }, ""
        structure, mfe = RNA.fold(rna_sequence)
        counts = Counter(structure)
        paired_fraction = (counts.get("(", 0) + counts.get(")", 0)) / max(len(structure), 1)
        loop_fraction = counts.get(".", 0) / max(len(structure), 1)
        transitions = sum(structure[index] != structure[index - 1] for index in range(1, len(structure))) / max(
            len(structure) - 1, 1
        )
        structure_entropy = _safe_entropy(counts.values()) / np.log2(max(len(counts), 2))
        return {
            "rna_mfe": _normalized(float(mfe), -350.0, 0.0),
            "rna_paired_fraction": float(paired_fraction),
            "rna_loop_fraction": float(loop_fraction),
            "rna_transition_rate": float(transitions),
            "rna_structure_entropy": float(structure_entropy),
        }, structure

    def _nucleotide_block(self, sequence: str, alphabet: Sequence[str]) -> Dict[str, float]:
        counts = Counter(sequence)
        total = max(len(sequence), 1)
        gc_count = counts.get("G", 0) + counts.get("C", 0)
        at_count = counts.get("A", 0) + counts.get("T", 0) + counts.get("U", 0)
        gc_content = gc_count / total
        entropy = _safe_entropy(counts.values()) / np.log2(len(alphabet))
        gc_skew = (counts.get("G", 0) - counts.get("C", 0)) / max(gc_count, 1)
        at_skew = (counts.get("A", 0) - counts.get("T", 0) - counts.get("U", 0)) / max(at_count, 1)
        codons = [
            sequence[index : index + 3]
            for index in range(0, len(sequence) - 2, 3)
            if len(sequence[index : index + 3]) == 3
        ]
        codon_counts = Counter(codons)
        periodicity = 0.0
        if len(sequence) >= 6:
            numeric = np.array([alphabet.index(char) for char in sequence if char in alphabet], dtype=np.float32)
            if numeric.size > 1:
                periodicity = float(np.std(np.diff(numeric))) / max(len(alphabet) - 1, 1)
        return {
            "seq_length_norm": _normalized(
                len(sequence), self.config.min_sequence_length, self.config.max_sequence_length
            ),
            "gc_content": float(gc_content),
            "entropy": float(entropy),
            "gc_skew": float(gc_skew),
            "at_skew": float(at_skew),
            "codon_entropy": _safe_entropy(codon_counts.values()) / np.log2(max(len(codon_counts), 2)),
            "periodicity": float(np.clip(periodicity, 0.0, 1.0)),
            **{f"freq_{symbol}": counts.get(symbol, 0) / total for symbol in alphabet},
        }

    def _vectorize(self, feature_map: Dict[str, float]) -> tuple[np.ndarray, List[str]]:
        feature_names = sorted(feature_map)
        values = np.array([feature_map[name] for name in feature_names], dtype=np.float32)
        if values.size >= self.config.embedding_dim:
            vector = values[: self.config.embedding_dim]
        else:
            summary = np.array(
                [
                    float(values.mean()) if values.size else 0.0,
                    float(values.std()) if values.size else 0.0,
                    float(values.min()) if values.size else 0.0,
                    float(values.max()) if values.size else 0.0,
                ],
                dtype=np.float32,
            )
            combined = np.concatenate([values, summary])
            vector = np.zeros(self.config.embedding_dim, dtype=np.float32)
            copy_len = min(vector.size, combined.size)
            vector[:copy_len] = combined[:copy_len]
        return vector, feature_names

    def _tonic_pc_hint(self, feature_map: Dict[str, float]) -> int:
        tonic_score = (
            0.22 * feature_map.get("freq_A", 0.25) * 0
            + 0.22 * feature_map.get("freq_C", 0.25) * 7
            + 0.22 * feature_map.get("freq_G", 0.25) * 2
            + 0.22 * (feature_map.get("freq_T", 0.0) + feature_map.get("freq_U", 0.0) + 1e-6) * 9
            + 0.04 * feature_map.get("protein_aromaticity", 0.0) * 11
            + 0.04 * _normalized(feature_map.get("protein_charge_density", 0.0), -0.3, 0.3) * 5
            + 0.04 * feature_map.get("rna_paired_fraction", 0.0) * 4
        )
        return int(round(tonic_score)) % 12

    def _music_control_profile(self, feature_map: Dict[str, float]) -> np.ndarray:
        tempo_likelihood = float(
            np.mean(
                [
                    feature_map.get("entropy", 0.0),
                    feature_map.get("codon_entropy", 0.0),
                    feature_map.get("periodicity", 0.0),
                    feature_map.get("protein_secondary_diversity", 0.0),
                    feature_map.get("rna_structure_entropy", 0.0),
                ]
            )
        )
        density = float(
            np.clip(
                0.35 * feature_map.get("protein_flexibility_mean", 0.0)
                + 0.25 * feature_map.get("gc_content", 0.0)
                + 0.20 * feature_map.get("protein_orf_coverage", 0.0)
                + 0.20 * feature_map.get("periodicity", 0.0),
                0.0,
                1.0,
            )
        )
        polyphony = float(
            np.clip(
                0.35 * feature_map.get("protein_secondary_diversity", 0.0)
                + 0.30 * feature_map.get("rna_paired_fraction", 0.0)
                + 0.20 * feature_map.get("protein_sheet", 0.0)
                + 0.15 * feature_map.get("protein_helix", 0.0),
                0.0,
                1.0,
            )
        )
        register = float(
            np.clip(
                0.40 * _normalized(feature_map.get("gc_skew", 0.0), -1.0, 1.0)
                + 0.30 * feature_map.get("protein_gravy", 0.0)
                + 0.30 * feature_map.get("protein_isoelectric_point", 0.5),
                0.0,
                1.0,
            )
        )
        harmony = float(
            np.clip(
                0.30 * (1.0 - feature_map.get("protein_instability", 0.5))
                + 0.20 * feature_map.get("protein_aromaticity", 0.0)
                + 0.20 * feature_map.get("rna_paired_fraction", 0.0)
                + 0.15 * feature_map.get("entropy", 0.0)
                + 0.15 * feature_map.get("protein_helix", 0.0),
                0.0,
                1.0,
            )
        )
        mode = float(
            np.clip(
                0.55 * feature_map.get("gc_content", 0.0)
                + 0.25 * (1.0 - feature_map.get("protein_instability", 0.5))
                + 0.20 * feature_map.get("rna_loop_fraction", 0.0),
                0.0,
                1.0,
            )
        )
        return np.array([tempo_likelihood, density, polyphony, register, harmony, mode], dtype=np.float32)

    def encode_sequence(self, sequence: str, sequence_id: str = "sequence_0") -> BioEncodingResult:
        sequence_type = self.infer_sequence_type(sequence)
        cleaned = self._clean_sequence(sequence, sequence_type)
        if len(cleaned) < self.config.min_sequence_length:
            raise ValueError(
                f"Sequence {sequence_id} is too short after cleaning: "
                f"{len(cleaned)} < {self.config.min_sequence_length}"
            )

        feature_map: Dict[str, float] = {}
        translated_protein = ""
        predicted_structure = ""

        if sequence_type in {"dna", "rna"}:
            alphabet = RNA_BASES if sequence_type == "rna" else NUCLEOTIDES
            feature_map.update(self._nucleotide_block(cleaned, alphabet))
            for k in self.config.kmer_sizes:
                kmer_distribution = self._kmer_distribution(cleaned, k, alphabet)
                for index, value in enumerate(kmer_distribution[: self.config.max_kmer_features]):
                    feature_map[f"kmer_{k}_{index:03d}"] = float(value)
            rna_sequence = cleaned.replace("T", "U")
            structure_block, predicted_structure = self._rna_structure_block(rna_sequence)
            feature_map.update(structure_block)
            if self.config.use_protein_features:
                translated_protein = self._longest_orf(cleaned)
                feature_map.update(self._protein_feature_block(translated_protein))
                feature_map["protein_orf_coverage"] = len(translated_protein) * 3 / max(len(cleaned), 1)
                feature_map.update(self._esm_embedding_block(translated_protein))
        else:
            feature_map["seq_length_norm"] = _normalized(
                len(cleaned), self.config.min_sequence_length, self.config.max_sequence_length
            )
            feature_map["entropy"] = _safe_entropy(Counter(cleaned).values()) / np.log2(len(AMINO_ACIDS))
            feature_map["periodicity"] = 0.0
            feature_map["gc_content"] = 0.5
            feature_map["gc_skew"] = 0.0
            feature_map["at_skew"] = 0.0
            feature_map["codon_entropy"] = 0.0
            translated_protein = cleaned
            feature_map.update(self._protein_feature_block(cleaned))
            feature_map.update(self._esm_embedding_block(cleaned))
            for acid in AMINO_ACIDS:
                feature_map.setdefault(f"freq_{acid}", feature_map.get(f"aa_{acid}", 0.0))

        vector, feature_names = self._vectorize(feature_map)
        return BioEncodingResult(
            sequence_id=sequence_id,
            sequence_type=sequence_type,
            cleaned_sequence=cleaned,
            vector=vector,
            control_profile=self._music_control_profile(feature_map),
            tonic_pc_hint=self._tonic_pc_hint(feature_map),
            feature_names=feature_names,
            feature_map=feature_map,
            translated_protein=translated_protein,
            predicted_structure=predicted_structure,
        )
