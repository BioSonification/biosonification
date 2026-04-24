"""Polyphonic music dataset tooling for the v2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import MusicDataConfig

try:
    from music21 import chord, converter, corpus, meter, note, stream, tempo
except ImportError:  # pragma: no cover - exercised during environment setup
    chord = None
    converter = None
    corpus = None
    meter = None
    note = None
    stream = None
    tempo = None


@dataclass
class NoteEvent:
    pitch: int
    offset: float
    duration: float
    velocity: int
    measure_number: int


@dataclass
class MusicSegment:
    segment_id: str
    source_path: str
    start_measure: int
    end_measure: int
    token_ids: List[int]
    control_token_ids: List[int]
    descriptor_vector: np.ndarray
    note_count: int
    mode: str
    metadata: Dict[str, float]


class PolyphonicMusicTokenizer:
    """Event tokenizer that preserves polyphony."""

    def __init__(self, config: Optional[MusicDataConfig] = None):
        self.config = config or MusicDataConfig()
        self.vocab = self._build_vocab()
        self.token_to_id = {token: index for index, token in enumerate(self.vocab)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["PAD"]
        self.bos_token_id = self.token_to_id["BOS"]
        self.eos_token_id = self.token_to_id["EOS"]
        self.sep_token_id = self.token_to_id["SEP"]
        self.max_time_shift = self.config.bars_per_segment * 4 * self.config.steps_per_beat * 2

    def _build_vocab(self) -> List[str]:
        vocab = ["PAD", "BOS", "EOS", "SEP"]
        vocab.extend(["CTRL_MODE_MAJOR", "CTRL_MODE_MINOR", "CTRL_MODE_OTHER"])
        for prefix in ("CTRL_TEMPO", "CTRL_DENSITY", "CTRL_POLY", "CTRL_REGISTER", "CTRL_HARMONY"):
            for bucket in range(self.config.descriptor_bins):
                vocab.append(f"{prefix}_{bucket}")
        for value in range(self.config.bars_per_segment * 4 * self.config.steps_per_beat * 2 + 1):
            vocab.append(f"TIME_{value}")
        for pitch in range(self.config.pitch_range_min, self.config.pitch_range_max + 1):
            vocab.append(f"NOTE_{pitch}")
        for duration in range(1, self.config.duration_bins + 1):
            vocab.append(f"DUR_{duration}")
        for velocity_bin in range(self.config.velocity_bins):
            vocab.append(f"VEL_{velocity_bin}")
        return vocab

    def control_tokens(self, descriptor_vector: np.ndarray, mode_name: str) -> List[int]:
        mode_token = "CTRL_MODE_OTHER"
        if mode_name == "major":
            mode_token = "CTRL_MODE_MAJOR"
        elif mode_name == "minor":
            mode_token = "CTRL_MODE_MINOR"
        tokens = [self.token_to_id[mode_token]]
        labels = ("CTRL_TEMPO", "CTRL_DENSITY", "CTRL_POLY", "CTRL_REGISTER", "CTRL_HARMONY")
        for label, value in zip(labels, descriptor_vector[:5]):
            bucket = int(np.clip(round(float(value) * (self.config.descriptor_bins - 1)), 0, self.config.descriptor_bins - 1))
            tokens.append(self.token_to_id[f"{label}_{bucket}"])
        return tokens

    def encode_events(
        self,
        events: Sequence[NoteEvent],
        descriptor_vector: np.ndarray,
        mode_name: str,
    ) -> Tuple[List[int], List[int]]:
        if not events:
            raise ValueError("Cannot encode an empty segment")
        tokens = [self.bos_token_id]
        control_token_ids = self.control_tokens(descriptor_vector, mode_name)
        tokens.extend(control_token_ids)
        tokens.append(self.sep_token_id)

        current_step = 0
        for event in sorted(events, key=lambda item: (item.offset, item.pitch, item.duration)):
            onset_step = max(0, int(round(event.offset * self.config.steps_per_beat)))
            delta = int(np.clip(onset_step - current_step, 0, self.max_time_shift))
            duration_steps = int(
                np.clip(round(event.duration * self.config.steps_per_beat), 1, self.config.duration_bins)
            )
            velocity_bucket = int(
                np.clip(event.velocity // max(1, 128 // self.config.velocity_bins), 0, self.config.velocity_bins - 1)
            )
            tokens.append(self.token_to_id[f"TIME_{delta}"])
            tokens.append(self.token_to_id[f"NOTE_{event.pitch}"])
            tokens.append(self.token_to_id[f"DUR_{duration_steps}"])
            tokens.append(self.token_to_id[f"VEL_{velocity_bucket}"])
            current_step = onset_step
            if len(tokens) >= self.config.max_events - 1:
                break

        tokens.append(self.eos_token_id)
        return tokens[: self.config.max_events], control_token_ids

    def decode_to_score(self, token_ids: Sequence[int], tempo_bpm: float = 96.0) -> "stream.Score":
        if stream is None or note is None or meter is None or tempo is None:
            raise ImportError("music21 is required to decode generated music.")

        score = stream.Score(id="generated_score")
        part = stream.Part(id="piano")
        part.insert(0, tempo.MetronomeMark(number=float(tempo_bpm)))
        part.insert(0, meter.TimeSignature("4/4"))

        current_step = 0
        index = 0
        while index < len(token_ids):
            token = self.id_to_token.get(int(token_ids[index]), "")
            if token == "EOS":
                break
            if not token.startswith("TIME_"):
                index += 1
                continue

            current_step += int(token.split("_")[1])
            if index + 3 >= len(token_ids):
                break
            note_token = self.id_to_token.get(int(token_ids[index + 1]), "")
            duration_token = self.id_to_token.get(int(token_ids[index + 2]), "")
            velocity_token = self.id_to_token.get(int(token_ids[index + 3]), "")
            if not note_token.startswith("NOTE_") or not duration_token.startswith("DUR_") or not velocity_token.startswith("VEL_"):
                index += 1
                continue

            pitch = int(note_token.split("_")[1])
            duration_steps = int(duration_token.split("_")[1])
            velocity_bucket = int(velocity_token.split("_")[1])
            quarter_length = duration_steps / self.config.steps_per_beat
            velocity_value = int((velocity_bucket + 0.5) * (128 / self.config.velocity_bins))

            midi_note = note.Note(pitch)
            midi_note.quarterLength = quarter_length
            midi_note.volume.velocity = max(1, min(127, velocity_value))
            midi_note.offset = current_step / self.config.steps_per_beat
            part.insert(midi_note.offset, midi_note)
            index += 4

        score.insert(0, part)
        return score

    def write_midi(self, token_ids: Sequence[int], output_path: str, tempo_bpm: float = 96.0) -> None:
        score = self.decode_to_score(token_ids, tempo_bpm=tempo_bpm)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        score.write("midi", fp=output_path)


class BioMusicPairDataset(Dataset):
    """Torch dataset over paired bio/music samples."""

    def __init__(self, samples: Sequence[dict], pad_token_id: int):
        self.samples = list(samples)
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        token_ids = torch.tensor(sample["token_ids"], dtype=torch.long)
        attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "bio_vector": torch.tensor(sample["bio_vector"], dtype=torch.float32),
            "descriptor_vector": torch.tensor(sample["descriptor_vector"], dtype=torch.float32),
            "pair_weight": torch.tensor(sample["pair_weight"], dtype=torch.float32),
            "segment_id": sample["segment_id"],
            "sequence_id": sample["sequence_id"],
        }

    def collate_fn(self, batch: Sequence[dict]) -> dict:
        max_len = max(item["input_ids"].size(0) for item in batch)
        input_ids = []
        attention_masks = []
        for item in batch:
            pad_amount = max_len - item["input_ids"].size(0)
            if pad_amount > 0:
                padding = torch.full((pad_amount,), self.pad_token_id, dtype=torch.long)
                mask_padding = torch.zeros((pad_amount,), dtype=torch.bool)
                input_ids.append(torch.cat([item["input_ids"], padding], dim=0))
                attention_masks.append(torch.cat([item["attention_mask"], mask_padding], dim=0))
            else:
                input_ids.append(item["input_ids"])
                attention_masks.append(item["attention_mask"])
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "bio_vector": torch.stack([item["bio_vector"] for item in batch]),
            "descriptor_vector": torch.stack([item["descriptor_vector"] for item in batch]),
            "pair_weight": torch.stack([item["pair_weight"] for item in batch]),
            "segment_id": [item["segment_id"] for item in batch],
            "sequence_id": [item["sequence_id"] for item in batch],
        }


def bootstrap_music21_corpus(output_dir: str, max_files: int = 96) -> List[Path]:
    """Export a compact polyphonic Bach corpus to local MIDI files."""

    if corpus is None:
        raise ImportError("music21 is required to bootstrap the fallback corpus.")

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(target_dir.glob("*.mid"))
    if len(existing) >= max_files:
        return existing[:max_files]

    created: List[Path] = []
    iterator = corpus.chorales.Iterator()
    for index, score in enumerate(iterator):
        if index >= max_files:
            break
        midi_path = target_dir / f"bach_chorale_{index:04d}.mid"
        if not midi_path.exists():
            score.write("midi", fp=str(midi_path))
        created.append(midi_path)
    return created


def _ensure_music21() -> None:
    if converter is None or note is None or chord is None or stream is None:
        raise ImportError("music21 is required for the v2 symbolic music pipeline.")


def _iter_score_files(midi_dirs: Sequence[str]) -> List[Path]:
    files: List[Path] = []
    extensions = ("*.mid", "*.midi", "*.xml", "*.mxl", "*.musicxml")
    for midi_dir in midi_dirs:
        base = Path(midi_dir)
        if not base.exists():
            continue
        for pattern in extensions:
            files.extend(base.rglob(pattern))
    deduped = sorted({path.resolve() for path in files})
    return deduped


def _first_measure_map(score: "stream.Score") -> Dict[int, float]:
    parts = list(score.parts) if hasattr(score, "parts") else []
    measure_container = parts[0] if parts else score
    mapping: Dict[int, float] = {}
    for measure in measure_container.recurse().getElementsByClass(stream.Measure):
        mapping[int(measure.number)] = float(measure.offset)
    if not mapping:
        max_offset = float(score.highestTime)
        total_measures = max(1, int(np.ceil(max_offset / 4.0)))
        mapping = {index + 1: index * 4.0 for index in range(total_measures + 1)}
    return mapping


def _extract_note_events(score: "stream.Score", config: MusicDataConfig) -> List[NoteEvent]:
    flat = score.flatten()
    events: List[NoteEvent] = []
    for element in flat.notes:
        measure_number = int(getattr(element, "measureNumber", 1) or 1)
        velocity = int(getattr(getattr(element, "volume", None), "velocity", 64) or 64)
        duration = float(element.quarterLength)
        offset = float(element.offset)
        if isinstance(element, note.Note):
            pitches = [int(element.pitch.midi)]
        elif isinstance(element, chord.Chord):
            pitches = [int(pitch_obj.midi) for pitch_obj in element.pitches]
        else:
            continue
        for midi_pitch in pitches:
            if config.pitch_range_min <= midi_pitch <= config.pitch_range_max:
                events.append(
                    NoteEvent(
                        pitch=midi_pitch,
                        offset=offset,
                        duration=max(0.25, duration),
                        velocity=velocity,
                        measure_number=measure_number,
                    )
                )
    return events


def _tempo_bpm(score: "stream.Score") -> float:
    marks = list(score.recurse().getElementsByClass(tempo.MetronomeMark))
    for mark in marks:
        if mark.number is not None:
            return float(mark.number)
    return 96.0


def _segment_descriptors(events: Sequence[NoteEvent], tempo_bpm: float, config: MusicDataConfig) -> Tuple[np.ndarray, str, Dict[str, float]]:
    if not events:
        raise ValueError("Cannot compute descriptors for an empty segment.")
    onsets: Dict[float, List[int]] = {}
    durations = []
    for event in events:
        onsets.setdefault(round(event.offset, 3), []).append(event.pitch)
        durations.append(event.duration)
    simultaneous = [len(pitches) for pitches in onsets.values()]
    multi_onset_ratio = sum(value > 1 for value in simultaneous) / max(len(simultaneous), 1)
    mean_pitch = float(np.mean([event.pitch for event in events]))
    density = len(events) / max(sum(durations), 1.0)

    consonant_classes = {0, 3, 4, 5, 7, 8, 9}
    consonant_pairs = 0
    total_pairs = 0
    pitch_classes = []
    for pitches in onsets.values():
        pitch_classes.extend([pitch % 12 for pitch in pitches])
        for left in range(len(pitches)):
            for right in range(left + 1, len(pitches)):
                total_pairs += 1
                interval_class = abs(pitches[right] - pitches[left]) % 12
                if interval_class in consonant_classes:
                    consonant_pairs += 1
    harmony = consonant_pairs / max(total_pairs, 1)
    pitch_class_counts = np.bincount(np.array(pitch_classes, dtype=np.int64), minlength=12)
    major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    major_score = float(np.dot(pitch_class_counts, major_profile))
    minor_score = float(np.dot(pitch_class_counts, minor_profile))
    mode_name = "major" if major_score >= minor_score else "minor"
    descriptor_vector = np.array(
        [
            np.clip((tempo_bpm - 48.0) / (168.0 - 48.0), 0.0, 1.0),
            np.clip(density / 3.0, 0.0, 1.0),
            np.clip(np.mean(simultaneous) / 4.0, 0.0, 1.0),
            np.clip((mean_pitch - config.pitch_range_min) / (config.pitch_range_max - config.pitch_range_min), 0.0, 1.0),
            np.clip(harmony, 0.0, 1.0),
            1.0 if mode_name == "major" else 0.0,
        ],
        dtype=np.float32,
    )
    metadata = {
        "tempo_bpm": float(tempo_bpm),
        "density": float(density),
        "polyphony_ratio": float(multi_onset_ratio),
        "mean_pitch": mean_pitch,
        "harmony": float(harmony),
    }
    return descriptor_vector, mode_name, metadata


def _segment_score(
    score_path: Path,
    tokenizer: PolyphonicMusicTokenizer,
    config: MusicDataConfig,
) -> List[MusicSegment]:
    _ensure_music21()
    score = converter.parse(str(score_path))
    score = score.makeMeasures(inPlace=False)
    measure_map = _first_measure_map(score)
    measure_numbers = sorted(measure_map)
    if not measure_numbers:
        return []
    all_events = _extract_note_events(score, config)
    if not all_events:
        return []

    tempo_bpm = _tempo_bpm(score)
    start_measure = min(measure_numbers)
    max_measure = max(measure_numbers)
    segments: List[MusicSegment] = []
    step = max(1, config.segment_hop_bars)
    for measure_index in range(start_measure, max_measure + 1, step):
        end_measure = measure_index + config.bars_per_segment
        segment_start = measure_map.get(measure_index, (measure_index - 1) * 4.0)
        segment_end = measure_map.get(end_measure, segment_start + config.bars_per_segment * 4.0)
        segment_events = [
            NoteEvent(
                pitch=event.pitch,
                offset=event.offset - segment_start,
                duration=min(event.duration, segment_end - event.offset),
                velocity=event.velocity,
                measure_number=event.measure_number,
            )
            for event in all_events
            if measure_index <= event.measure_number < end_measure and event.offset < segment_end
        ]
        if len(segment_events) < config.min_notes_per_segment:
            continue
        descriptor_vector, mode_name, metadata = _segment_descriptors(segment_events, tempo_bpm, config)
        if metadata["polyphony_ratio"] < config.min_polyphony_ratio:
            continue
        token_ids, control_token_ids = tokenizer.encode_events(segment_events, descriptor_vector, mode_name)
        segment_id = f"{score_path.stem}_m{measure_index:03d}"
        segments.append(
            MusicSegment(
                segment_id=segment_id,
                source_path=str(score_path),
                start_measure=measure_index,
                end_measure=end_measure,
                token_ids=token_ids,
                control_token_ids=control_token_ids,
                descriptor_vector=descriptor_vector,
                note_count=len(segment_events),
                mode=mode_name,
                metadata=metadata,
            )
        )
    return segments


def load_music_corpus(
    config: Optional[MusicDataConfig] = None,
    tokenizer: Optional[PolyphonicMusicTokenizer] = None,
) -> Tuple[List[MusicSegment], PolyphonicMusicTokenizer]:
    """Load or bootstrap a polyphonic symbolic music corpus."""

    music_config = config or MusicDataConfig()
    _ensure_music21()
    tokenizer = tokenizer or PolyphonicMusicTokenizer(music_config)

    midi_dirs = list(music_config.midi_dirs)
    files = _iter_score_files(midi_dirs)
    if not files and music_config.use_music21_corpus_fallback:
        fallback_dir = Path(midi_dirs[0]) if midi_dirs else Path("data/midi/polyphonic_music21")
        bootstrap_music21_corpus(str(fallback_dir), max_files=music_config.max_music21_files)
        files = _iter_score_files([str(fallback_dir)])

    segments: List[MusicSegment] = []
    for score_path in files:
        segments.extend(_segment_score(score_path, tokenizer, music_config))
    if not segments:
        raise ValueError(
            "No valid polyphonic music segments were found. "
            "Provide a MIDI corpus or enable the music21 fallback corpus."
        )
    return segments, tokenizer
