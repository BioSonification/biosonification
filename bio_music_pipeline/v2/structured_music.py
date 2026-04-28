"""Hierarchical chord-plus-melody music representation for the v2 pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .config import MusicDataConfig
from .corpus import bootstrap_music21_corpus, iter_score_files

try:
    from music21 import chord, converter, corpus, key, meter, note, stream, tempo
except ImportError:  # pragma: no cover - checked by runtime bootstrap
    chord = None
    converter = None
    corpus = None
    key = None
    meter = None
    note = None
    stream = None
    tempo = None

try:
    import mido
except ImportError:  # pragma: no cover
    mido = None


CHORD_QUALITIES = (
    "maj",
    "min",
    "dim",
    "aug",
    "dom7",
    "maj7",
    "min7",
    "sus2",
    "sus4",
    "power",
    "other",
)

QUALITY_INTERVALS: Dict[str, Tuple[int, ...]] = {
    "maj": (0, 4, 7),
    "min": (0, 3, 7),
    "dim": (0, 3, 6),
    "aug": (0, 4, 8),
    "dom7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "power": (0, 7),
    "other": (0, 4, 7),
}

PITCH_CLASS_BY_NAME = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

PITCH_CLASS_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


@dataclass
class HarmonyBar:
    bar_index: int
    root_pc: int
    quality: str
    hold: bool
    key_tonic_pc: int
    key_mode: str


@dataclass
class MelodyEvent:
    bar_index: int
    position_step: int
    duration_steps: int
    relative_pc: int
    octave: int


@dataclass
class StructuredMusicSegment:
    segment_id: str
    source_path: str
    start_measure: int
    end_measure: int
    harmony_bars: List[HarmonyBar]
    melody_events: List[MelodyEvent]
    harmony_token_ids: List[int]
    harmony_prefix_ids: List[int]
    melody_token_ids: List[int]
    melody_prefix_ids: List[int]
    descriptor_vector: np.ndarray
    tempo_bpm: float
    key_tonic_pc: int
    key_mode: str


class HarmonyTokenizer:
    """Tokenizer for bar-level chord progressions."""

    def __init__(self, config: Optional[MusicDataConfig] = None):
        self.config = config or MusicDataConfig()
        self.vocab = self._build_vocab()
        self.token_to_id = {token: index for index, token in enumerate(self.vocab)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["PAD"]
        self.bos_token_id = self.token_to_id["BOS"]
        self.eos_token_id = self.token_to_id["EOS"]
        self.sep_token_id = self.token_to_id["SEP"]

    def _build_vocab(self) -> List[str]:
        vocab = ["PAD", "BOS", "EOS", "SEP", "BAR", "HOLD"]
        vocab.extend(["CTRL_MODE_MAJOR", "CTRL_MODE_MINOR", "CTRL_MODE_OTHER"])
        for tonic in range(12):
            vocab.append(f"CTRL_TONIC_{tonic}")
        for prefix in ("CTRL_DENSITY", "CTRL_REGISTER", "CTRL_HARMONY", "CTRL_CHANGE"):
            for value in range(self.config.descriptor_bins):
                vocab.append(f"{prefix}_{value}")
        for root_pc in range(12):
            vocab.append(f"ROOT_{root_pc}")
        for quality in CHORD_QUALITIES:
            vocab.append(f"QUAL_{quality}")
        return vocab

    def control_tokens(self, descriptor_vector: np.ndarray, tonic_pc_hint: int, mode_name: str) -> List[int]:
        mode_token = "CTRL_MODE_OTHER"
        if mode_name == "major":
            mode_token = "CTRL_MODE_MAJOR"
        elif mode_name == "minor":
            mode_token = "CTRL_MODE_MINOR"
        tokens = [self.token_to_id[mode_token]]
        tonic_bucket = int(np.clip(tonic_pc_hint, 0, 11))
        tokens.append(self.token_to_id[f"CTRL_TONIC_{tonic_bucket}"])
        value_indices = (1, 3, 4, 2)
        labels = ("CTRL_DENSITY", "CTRL_REGISTER", "CTRL_HARMONY", "CTRL_CHANGE")
        for label, index in zip(labels, value_indices):
            bucket = int(np.clip(round(float(descriptor_vector[index]) * (self.config.descriptor_bins - 1)), 0, self.config.descriptor_bins - 1))
            tokens.append(self.token_to_id[f"{label}_{bucket}"])
        return tokens

    def encode_progression(
        self,
        harmony_bars: Sequence[HarmonyBar],
        descriptor_vector: np.ndarray,
        tonic_pc_hint: int,
        mode_name: str,
    ) -> Tuple[List[int], List[int]]:
        control_ids = self.control_tokens(descriptor_vector, tonic_pc_hint, mode_name)
        tokens = [self.bos_token_id, *control_ids, self.sep_token_id]
        for harmony_bar in harmony_bars:
            tokens.append(self.token_to_id["BAR"])
            if harmony_bar.hold and self.config.chord_hold_token_enabled:
                tokens.append(self.token_to_id["HOLD"])
            else:
                tokens.append(self.token_to_id[f"ROOT_{harmony_bar.root_pc}"])
                quality = harmony_bar.quality if harmony_bar.quality in CHORD_QUALITIES else "other"
                tokens.append(self.token_to_id[f"QUAL_{quality}"])
        tokens.append(self.eos_token_id)
        return tokens, control_ids

    def decode_progression(self, token_ids: Sequence[int], num_bars: int) -> List[HarmonyBar]:
        bars: List[HarmonyBar] = []
        current_root = 0
        current_quality = "maj"
        index = 0
        while index < len(token_ids) and len(bars) < num_bars:
            token = self.id_to_token.get(int(token_ids[index]), "")
            if token == "EOS":
                break
            if token != "BAR":
                index += 1
                continue
            if index + 1 >= len(token_ids):
                break
            next_token = self.id_to_token.get(int(token_ids[index + 1]), "")
            if next_token == "HOLD":
                bars.append(HarmonyBar(len(bars), current_root, current_quality, True, 0, "major"))
                index += 2
                continue
            if not next_token.startswith("ROOT_") or index + 2 >= len(token_ids):
                index += 1
                continue
            root_pc = int(next_token.split("_")[1])
            quality_token = self.id_to_token.get(int(token_ids[index + 2]), "")
            quality = quality_token.replace("QUAL_", "") if quality_token.startswith("QUAL_") else "other"
            current_root = root_pc
            current_quality = quality
            bars.append(HarmonyBar(len(bars), root_pc, quality, False, 0, "major"))
            index += 3
        while len(bars) < num_bars:
            bars.append(HarmonyBar(len(bars), current_root, current_quality, True, 0, "major"))
        return bars


class MelodyTokenizer:
    """Tokenizer for monophonic melody conditioned on harmony prefixes."""

    def __init__(self, config: Optional[MusicDataConfig] = None):
        self.config = config or MusicDataConfig()
        self.vocab = self._build_vocab()
        self.token_to_id = {token: index for index, token in enumerate(self.vocab)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}
        self.pad_token_id = self.token_to_id["PAD"]
        self.bos_token_id = self.token_to_id["BOS"]
        self.eos_token_id = self.token_to_id["EOS"]
        self.sep_token_id = self.token_to_id["SEP"]

    def _build_vocab(self) -> List[str]:
        vocab = ["PAD", "BOS", "EOS", "SEP", "BAR"]
        vocab.extend(["CTRL_MODE_MAJOR", "CTRL_MODE_MINOR", "CTRL_MODE_OTHER"])
        for prefix in ("CTRL_DENSITY", "CTRL_REGISTER", "CTRL_HARMONY", "CTRL_CHANGE"):
            for value in range(self.config.descriptor_bins):
                vocab.append(f"{prefix}_{value}")
        for tonic in range(12):
            vocab.append(f"CTRL_TONIC_{tonic}")
        vocab.append("H_BAR")
        vocab.append("H_HOLD")
        for root_pc in range(12):
            vocab.append(f"H_ROOT_{root_pc}")
        for quality in CHORD_QUALITIES:
            vocab.append(f"H_QUAL_{quality}")
        for position in range(self.config.steps_per_bar):
            vocab.append(f"POS_{position}")
        for relative_pc in range(12):
            vocab.append(f"RELPC_{relative_pc}")
        for octave in range(self.config.melody_octave_min, self.config.melody_octave_max + 1):
            vocab.append(f"OCT_{octave}")
        for duration in range(1, self.config.steps_per_bar + 1):
            vocab.append(f"DUR_{duration}")
        return vocab

    def control_tokens(self, descriptor_vector: np.ndarray, tonic_pc_hint: int, mode_name: str) -> List[int]:
        mode_token = "CTRL_MODE_OTHER"
        if mode_name == "major":
            mode_token = "CTRL_MODE_MAJOR"
        elif mode_name == "minor":
            mode_token = "CTRL_MODE_MINOR"
        tokens = [self.token_to_id[mode_token]]
        tokens.append(self.token_to_id[f"CTRL_TONIC_{tonic_pc_hint % 12}"])
        value_indices = (1, 3, 4, 2)
        labels = ("CTRL_DENSITY", "CTRL_REGISTER", "CTRL_HARMONY", "CTRL_CHANGE")
        for label, index in zip(labels, value_indices):
            bucket = int(np.clip(round(float(descriptor_vector[index]) * (self.config.descriptor_bins - 1)), 0, self.config.descriptor_bins - 1))
            tokens.append(self.token_to_id[f"{label}_{bucket}"])
        return tokens

    def harmony_prefix_tokens(self, harmony_bars: Sequence[HarmonyBar]) -> List[str]:
        tokens: List[str] = []
        for harmony_bar in harmony_bars:
            tokens.append("H_BAR")
            if harmony_bar.hold and self.config.chord_hold_token_enabled:
                tokens.append("H_HOLD")
            else:
                tokens.append(f"H_ROOT_{harmony_bar.root_pc}")
                quality = harmony_bar.quality if harmony_bar.quality in CHORD_QUALITIES else "other"
                tokens.append(f"H_QUAL_{quality}")
        return tokens

    def encode_melody(
        self,
        melody_events: Sequence[MelodyEvent],
        harmony_bars: Sequence[HarmonyBar],
        descriptor_vector: np.ndarray,
        tonic_pc_hint: int,
        mode_name: str,
    ) -> Tuple[List[int], List[int]]:
        control_ids = self.control_tokens(descriptor_vector, tonic_pc_hint, mode_name)
        prefix_tokens = [self.bos_token_id, *control_ids]
        prefix_tokens.extend(self.token_to_id[token] for token in self.harmony_prefix_tokens(harmony_bars))
        prefix_tokens.append(self.sep_token_id)
        tokens = list(prefix_tokens)
        current_bar = -1
        for event in sorted(melody_events, key=lambda item: (item.bar_index, item.position_step, item.octave, item.relative_pc)):
            while current_bar < event.bar_index:
                current_bar += 1
                tokens.append(self.token_to_id["BAR"])
            tokens.append(self.token_to_id[f"POS_{event.position_step}"])
            tokens.append(self.token_to_id[f"RELPC_{event.relative_pc}"])
            tokens.append(self.token_to_id[f"OCT_{event.octave}"])
            tokens.append(self.token_to_id[f"DUR_{event.duration_steps}"])
        tokens.append(self.eos_token_id)
        return tokens, prefix_tokens

    def decode_melody(
        self,
        token_ids: Sequence[int],
        harmony_bars: Sequence[HarmonyBar],
        tonic_pc: int,
    ) -> List[Tuple[int, int, int, int]]:
        events: List[Tuple[int, int, int, int]] = []
        total_steps = max(len(harmony_bars) * self.config.steps_per_bar, self.config.steps_per_bar)
        bar_index = -1
        index = 0
        while index < len(token_ids):
            token = self.id_to_token.get(int(token_ids[index]), "")
            if token == "EOS":
                break
            if token == "BAR":
                bar_index += 1
                if bar_index >= len(harmony_bars):
                    break
                index += 1
                continue
            if not token.startswith("POS_") or index + 3 >= len(token_ids):
                index += 1
                continue
            pos_token = token
            rel_pc_token = self.id_to_token.get(int(token_ids[index + 1]), "")
            oct_token = self.id_to_token.get(int(token_ids[index + 2]), "")
            dur_token = self.id_to_token.get(int(token_ids[index + 3]), "")
            if not rel_pc_token.startswith("RELPC_") or not oct_token.startswith("OCT_") or not dur_token.startswith("DUR_"):
                index += 1
                continue
            if bar_index < 0 or bar_index >= len(harmony_bars):
                index += 4
                continue
            position_step = int(pos_token.split("_")[1])
            relative_pc = int(rel_pc_token.split("_")[1])
            octave = int(oct_token.split("_")[1])
            duration_steps = int(dur_token.split("_")[1])
            active_root = tonic_pc
            if 0 <= bar_index < len(harmony_bars):
                active_root = harmony_bars[bar_index].root_pc
            pitch = 12 * (octave + 1) + ((active_root + relative_pc) % 12)
            onset_step = bar_index * self.config.steps_per_bar + position_step
            if onset_step >= total_steps:
                index += 4
                continue
            duration_steps = int(np.clip(duration_steps, 1, total_steps - onset_step))
            events.append((onset_step, duration_steps, pitch, bar_index))
            index += 4
        return _normalize_decoded_melody_events(events, total_steps)


def _normalize_decoded_melody_events(
    events: Sequence[Tuple[int, int, int, int]],
    total_steps: int,
) -> List[Tuple[int, int, int, int]]:
    if not events:
        return []

    by_onset: Dict[int, List[Tuple[int, int, int, int]]] = {}
    for onset_step, duration_steps, pitch, bar_index in events:
        if onset_step < 0 or onset_step >= total_steps:
            continue
        by_onset.setdefault(int(onset_step), []).append((int(onset_step), int(duration_steps), int(pitch), int(bar_index)))

    deduped = [
        max(group, key=lambda item: (item[2], item[1]))
        for _, group in sorted(by_onset.items())
    ]

    normalized: List[Tuple[int, int, int, int]] = []
    for index, (onset_step, duration_steps, pitch, bar_index) in enumerate(deduped):
        next_onset = deduped[index + 1][0] if index + 1 < len(deduped) else total_steps
        max_duration = max(1, next_onset - onset_step)
        clipped_duration = int(np.clip(duration_steps, 1, max_duration))
        if onset_step + clipped_duration > total_steps:
            clipped_duration = total_steps - onset_step
        if clipped_duration <= 0:
            continue
        normalized.append((onset_step, clipped_duration, pitch, bar_index))
    return normalized


def _tempo_bpm(score: "stream.Score") -> float:
    marks = list(score.recurse().getElementsByClass(tempo.MetronomeMark))
    for mark in marks:
        if mark.number is not None:
            return float(mark.number)
    return 96.0


def _ensure_music21() -> None:
    if converter is None or note is None or chord is None or stream is None:
        raise ImportError("music21 is required for the v2 structured music pipeline.")


def _pick_melody_part(score: "stream.Score", config: MusicDataConfig) -> Optional["stream.Part"]:
    parts = list(score.parts)
    if not parts:
        return None
    preferred_names = ("melody", "soprano", "lead", "voice")
    if config.prefer_named_melody_parts:
        for part in parts:
            candidate = f"{part.partName or ''} {part.id or ''}".lower()
            if any(label in candidate for label in preferred_names):
                return part

    def part_score(part: "stream.Part") -> Tuple[float, float]:
        pitches = []
        n_notes = 0
        for element in part.flatten().notes:
            n_notes += 1
            if isinstance(element, note.Note):
                pitches.append(int(element.pitch.midi))
            elif isinstance(element, chord.Chord):
                pitches.extend(int(item.midi) for item in element.pitches)
        return (float(np.mean(pitches)) if pitches else 0.0, -float(n_notes))

    return max(parts, key=part_score)


def _classify_chord_symbol(chord_obj: "chord.Chord") -> Tuple[int, str]:
    root = chord_obj.root()
    root_pc = int(root.pitchClass) if root is not None else 0
    if chord_obj.isMajorTriad():
        return root_pc, "maj"
    if chord_obj.isMinorTriad():
        return root_pc, "min"
    if chord_obj.isDiminishedTriad():
        return root_pc, "dim"
    if chord_obj.isAugmentedTriad():
        return root_pc, "aug"
    common_name = (chord_obj.commonName or "").lower()
    pitch_classes = sorted({pitch_item.pitchClass for pitch_item in chord_obj.pitches})
    intervals = sorted({(pitch_class - root_pc) % 12 for pitch_class in pitch_classes})
    if 10 in intervals and 4 in intervals and 7 in intervals:
        return root_pc, "dom7"
    if 11 in intervals and 4 in intervals and 7 in intervals:
        return root_pc, "maj7"
    if 10 in intervals and 3 in intervals and 7 in intervals:
        return root_pc, "min7"
    if intervals == [0, 2, 7]:
        return root_pc, "sus2"
    if intervals == [0, 5, 7]:
        return root_pc, "sus4"
    if intervals == [0, 7]:
        return root_pc, "power"
    if "dominant" in common_name:
        return root_pc, "dom7"
    if "minor seventh" in common_name:
        return root_pc, "min7"
    if "major seventh" in common_name:
        return root_pc, "maj7"
    return root_pc, "other"


def _measure_offsets(score: "stream.Score") -> Dict[int, float]:
    parts = list(score.parts)
    measure_container = parts[0] if parts else score
    offsets: Dict[int, float] = {}
    for measure in measure_container.recurse().getElementsByClass(stream.Measure):
        offsets[int(measure.number)] = float(measure.offset)
    if not offsets:
        total_bars = max(1, int(np.ceil(float(score.highestTime) / 4.0)))
        offsets = {index + 1: index * 4.0 for index in range(total_bars + 1)}
    return offsets


def _extract_harmony_bars(
    score: "stream.Score",
    start_measure: int,
    end_measure: int,
    key_signature: "key.Key",
) -> List[HarmonyBar]:
    chordified = score.chordify()
    selected: List[HarmonyBar] = []
    previous_root = int(key_signature.tonic.pitchClass)
    previous_quality = "maj" if key_signature.mode == "major" else "min"
    for offset_bar, measure_number in enumerate(range(start_measure, end_measure)):
        measure_obj = chordified.measure(measure_number)
        selected_chord: Optional["chord.Chord"] = None
        if measure_obj is not None:
            candidates = list(measure_obj.recurse().getElementsByClass(chord.Chord))
            if candidates:
                selected_chord = max(candidates, key=lambda item: float(item.quarterLength))
        if selected_chord is None:
            selected.append(
                HarmonyBar(
                    bar_index=offset_bar,
                    root_pc=previous_root,
                    quality=previous_quality,
                    hold=True,
                    key_tonic_pc=int(key_signature.tonic.pitchClass),
                    key_mode=str(key_signature.mode),
                )
            )
            continue
        root_pc, quality = _classify_chord_symbol(selected_chord)
        hold = root_pc == previous_root and quality == previous_quality
        previous_root = root_pc
        previous_quality = quality
        selected.append(
            HarmonyBar(
                bar_index=offset_bar,
                root_pc=root_pc,
                quality=quality,
                hold=hold,
                key_tonic_pc=int(key_signature.tonic.pitchClass),
                key_mode=str(key_signature.mode),
            )
        )
    return selected


def _extract_melody_events(
    melody_part: "stream.Part",
    start_measure: int,
    end_measure: int,
    harmony_bars: Sequence[HarmonyBar],
    config: MusicDataConfig,
) -> List[MelodyEvent]:
    events: List[MelodyEvent] = []
    for element in melody_part.flatten().notes:
        measure_number = int(getattr(element, "measureNumber", 1) or 1)
        if measure_number < start_measure or measure_number >= end_measure:
            continue
        if isinstance(element, chord.Chord):
            pitch_obj = max(element.pitches, key=lambda item: item.midi)
            midi_pitch = int(pitch_obj.midi)
            octave_value = pitch_obj.octave or 4
        else:
            midi_pitch = int(element.pitch.midi)
            octave_value = element.pitch.octave or 4
        local_bar = measure_number - start_measure
        if not (0 <= local_bar < len(harmony_bars)):
            continue
        parent_measure = element.getContextByClass(stream.Measure)
        if parent_measure is not None:
            offset_within_measure = float(element.getOffsetInHierarchy(parent_measure))
        else:
            element_offset = float(element.offset)
            offset_within_measure = max(0.0, element_offset - local_bar * 4.0)
        position_step = int(np.clip(round(offset_within_measure * config.steps_per_beat), 0, config.steps_per_bar - 1))
        duration_steps = int(np.clip(round(float(element.quarterLength) * config.steps_per_beat), 1, config.steps_per_bar))
        active_root = harmony_bars[local_bar].root_pc
        relative_pc = (midi_pitch - active_root) % 12
        octave = int(np.clip(octave_value, config.melody_octave_min, config.melody_octave_max))
        events.append(
            MelodyEvent(
                bar_index=local_bar,
                position_step=position_step,
                duration_steps=duration_steps,
                relative_pc=relative_pc,
                octave=octave,
            )
        )
    events.sort(key=lambda item: (item.bar_index, item.position_step, item.octave))
    monophonic: List[MelodyEvent] = []
    seen_positions = set()
    for event in events:
        signature = (event.bar_index, event.position_step)
        if signature in seen_positions:
            continue
        seen_positions.add(signature)
        monophonic.append(event)
    return monophonic


def _segment_descriptor_vector(
    harmony_bars: Sequence[HarmonyBar],
    melody_events: Sequence[MelodyEvent],
    key_signature: "key.Key",
    tempo_bpm: float,
    config: MusicDataConfig,
) -> np.ndarray:
    if not melody_events:
        raise ValueError("Structured segment requires at least one melody event.")
    change_count = sum(
        1 for index in range(1, len(harmony_bars))
        if not harmony_bars[index].hold and (
            harmony_bars[index].root_pc != harmony_bars[index - 1].root_pc
            or harmony_bars[index].quality != harmony_bars[index - 1].quality
        )
    )
    note_density = len(melody_events) / max(len(harmony_bars), 1)
    mean_octave = np.mean([event.octave for event in melody_events])
    quality_complexity = np.mean(
        [
            1.0 if bar.quality in {"dom7", "maj7", "min7", "other"} else 0.5 if bar.quality in {"dim", "aug", "sus2", "sus4"} else 0.25
            for bar in harmony_bars
        ]
    )
    chord_tone_ratio = np.mean([1.0 if event.relative_pc in {0, 3, 4, 7, 10, 11} else 0.0 for event in melody_events])
    return np.array(
        [
            np.clip((tempo_bpm - 48.0) / 120.0, 0.0, 1.0),
            np.clip(note_density / 4.0, 0.0, 1.0),
            np.clip(change_count / max(len(harmony_bars) - 1, 1), 0.0, 1.0),
            np.clip((mean_octave - config.melody_octave_min) / max(config.melody_octave_max - config.melody_octave_min, 1), 0.0, 1.0),
            np.clip((quality_complexity + chord_tone_ratio) / 2.0, 0.0, 1.0),
            1.0 if str(key_signature.mode) == "major" else 0.0,
        ],
        dtype=np.float32,
    )


def _structured_segments_from_score(
    score_path: Path,
    harmony_tokenizer: HarmonyTokenizer,
    melody_tokenizer: MelodyTokenizer,
    config: MusicDataConfig,
) -> List[StructuredMusicSegment]:
    score = converter.parse(str(score_path))
    score = score.makeMeasures(inPlace=False)
    melody_part = _pick_melody_part(score, config)
    if melody_part is None:
        return []
    measure_map = _measure_offsets(score)
    measure_numbers = sorted(measure_map)
    if not measure_numbers:
        return []
    tempo_bpm = _tempo_bpm(score)
    try:
        key_signature = score.analyze("key")
    except Exception:
        key_signature = key.Key("C")
    start_measure = min(measure_numbers)
    max_measure = max(measure_numbers)
    segments: List[StructuredMusicSegment] = []
    for measure_index in range(start_measure, max_measure + 1, max(1, config.segment_hop_bars)):
        end_measure = measure_index + config.bars_per_segment
        if end_measure > max_measure + 1:
            continue
        harmony_bars = _extract_harmony_bars(score, measure_index, end_measure, key_signature)
        melody_events = _extract_melody_events(melody_part, measure_index, end_measure, harmony_bars, config)
        if len(melody_events) < config.min_notes_per_segment:
            continue
        descriptor_vector = _segment_descriptor_vector(harmony_bars, melody_events, key_signature, tempo_bpm, config)
        harmony_token_ids, harmony_prefix_ids = harmony_tokenizer.encode_progression(
            harmony_bars,
            descriptor_vector,
            int(key_signature.tonic.pitchClass),
            str(key_signature.mode),
        )
        melody_token_ids, melody_prefix_ids = melody_tokenizer.encode_melody(
            melody_events,
            harmony_bars,
            descriptor_vector,
            int(key_signature.tonic.pitchClass),
            str(key_signature.mode),
        )
        segments.append(
            StructuredMusicSegment(
                segment_id=f"{score_path.stem}_m{measure_index:03d}",
                source_path=str(score_path),
                start_measure=measure_index,
                end_measure=end_measure,
                harmony_bars=harmony_bars,
                melody_events=melody_events,
                harmony_token_ids=harmony_token_ids,
                harmony_prefix_ids=harmony_prefix_ids,
                melody_token_ids=melody_token_ids,
                melody_prefix_ids=melody_prefix_ids,
                descriptor_vector=descriptor_vector,
                tempo_bpm=tempo_bpm,
                key_tonic_pc=int(key_signature.tonic.pitchClass),
                key_mode=str(key_signature.mode),
            )
        )
    return segments


def _parse_pop909_chord_label(label: str, fallback_root: int, fallback_quality: str) -> Tuple[int, str]:
    if label == "N" or not label:
        return fallback_root, fallback_quality
    chord_name = label.split("/", 1)[0]
    if ":" not in chord_name:
        return fallback_root, fallback_quality
    root_name, quality_name = chord_name.split(":", 1)
    root_pc = PITCH_CLASS_BY_NAME.get(root_name, fallback_root)
    quality_name = quality_name.strip()
    if quality_name in {"maj", "min", "dim", "aug", "maj7", "min7", "sus2", "sus4"}:
        quality = quality_name
    elif quality_name in {"7", "sus4(b7)"}:
        quality = "dom7"
    elif quality_name in {"maj6"}:
        quality = "maj"
    elif quality_name in {"min6"}:
        quality = "min"
    elif quality_name in {"dim7", "hdim7"}:
        quality = "dim"
    else:
        quality = "other"
    return root_pc, quality


def _read_pop909_beats(song_dir: Path) -> Optional[Tuple[np.ndarray, List[int]]]:
    beat_path = song_dir / "beat_midi.txt"
    if not beat_path.exists():
        return None
    beat_times = []
    downbeats = []
    for line in beat_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        beat_times.append(float(parts[0]))
        if float(parts[2]) > 0.5:
            downbeats.append(len(beat_times) - 1)
    if len(beat_times) < 4 or len(downbeats) < 2:
        return None
    return np.asarray(beat_times, dtype=np.float64), downbeats


def _read_pop909_chords(song_dir: Path) -> List[Tuple[float, float, str]]:
    chord_path = song_dir / "chord_midi.txt"
    if not chord_path.exists():
        return []
    chords = []
    for line in chord_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            chords.append((float(parts[0]), float(parts[1]), parts[2]))
    return chords


def _midi_tempo_events(mid: "mido.MidiFile") -> List[Tuple[int, int]]:
    events = [(0, 500000)]
    for track in mid.tracks:
        absolute_tick = 0
        for message in track:
            absolute_tick += message.time
            if message.type == "set_tempo":
                events.append((absolute_tick, int(message.tempo)))
    return sorted(events, key=lambda item: item[0])


def _tick_to_seconds(tick: int, tempo_events: Sequence[Tuple[int, int]], ticks_per_beat: int) -> float:
    seconds = 0.0
    previous_tick = 0
    tempo_value = tempo_events[0][1]
    for event_tick, event_tempo in tempo_events[1:]:
        if tick <= event_tick:
            break
        seconds += mido.tick2second(event_tick - previous_tick, ticks_per_beat, tempo_value)
        previous_tick = event_tick
        tempo_value = event_tempo
    seconds += mido.tick2second(tick - previous_tick, ticks_per_beat, tempo_value)
    return float(seconds)


def _pop909_melody_notes(midi_path: Path) -> List[Tuple[float, float, int]]:
    if mido is None:
        return []
    mid = mido.MidiFile(str(midi_path))
    tempo_events = _midi_tempo_events(mid)
    melody_track = None
    fallback_track = None
    fallback_note_count = -1
    for track in mid.tracks:
        track_name = ""
        note_count = 0
        for message in track:
            if message.type == "track_name":
                track_name = message.name.upper()
            elif message.type == "note_on" and message.velocity > 0:
                note_count += 1
        if "MELODY" in track_name:
            melody_track = track
            break
        if note_count > fallback_note_count:
            fallback_track = track
            fallback_note_count = note_count
    track = melody_track or fallback_track
    if track is None:
        return []

    active: Dict[Tuple[int, int], List[int]] = {}
    notes_out: List[Tuple[float, float, int]] = []
    absolute_tick = 0
    for message in track:
        absolute_tick += message.time
        if message.type == "note_on" and message.velocity > 0:
            active.setdefault((message.channel, message.note), []).append(absolute_tick)
        elif message.type in {"note_off", "note_on"}:
            key_value = (message.channel, message.note)
            starts = active.get(key_value)
            if starts:
                start_tick = starts.pop(0)
                if absolute_tick > start_tick:
                    notes_out.append(
                        (
                            _tick_to_seconds(start_tick, tempo_events, mid.ticks_per_beat),
                            _tick_to_seconds(absolute_tick, tempo_events, mid.ticks_per_beat),
                            int(message.note),
                        )
                    )
    notes_out.sort(key=lambda item: (item[0], item[2]))
    return notes_out


def _time_to_beat(time_seconds: float, beat_times: np.ndarray) -> float:
    return float(np.interp(time_seconds, beat_times, np.arange(len(beat_times), dtype=np.float64)))


def _active_pop909_label(chords: Sequence[Tuple[float, float, str]], time_seconds: float) -> str:
    for start, end, label in chords:
        if start <= time_seconds < end:
            return label
    return chords[-1][2] if chords else "N"


def _pop909_segments_from_score(
    score_path: Path,
    harmony_tokenizer: HarmonyTokenizer,
    melody_tokenizer: MelodyTokenizer,
    config: MusicDataConfig,
) -> Optional[List[StructuredMusicSegment]]:
    song_dir = score_path.parent
    if song_dir.name == "versions":
        return []
    if not (song_dir / "beat_midi.txt").exists() or not (song_dir / "chord_midi.txt").exists():
        return None
    beat_data = _read_pop909_beats(song_dir)
    if beat_data is None:
        return []
    beat_times, downbeats = beat_data
    chords = _read_pop909_chords(song_dir)
    melody_notes = _pop909_melody_notes(score_path)
    if not melody_notes:
        return []

    beat_diffs = np.diff(beat_times)
    tempo_bpm = float(60.0 / np.median(beat_diffs)) if beat_diffs.size else 120.0
    segments: List[StructuredMusicSegment] = []
    hop = max(1, config.segment_hop_bars)
    bars_per_segment = max(1, config.bars_per_segment)
    for bar_start_index in range(0, len(downbeats) - bars_per_segment, hop):
        start_beat_index = downbeats[bar_start_index]
        end_beat_index = downbeats[bar_start_index + bars_per_segment]
        start_time = float(beat_times[start_beat_index])
        end_time = float(beat_times[end_beat_index])
        harmony_bars: List[HarmonyBar] = []
        previous_root = 0
        previous_quality = "maj"
        for local_bar in range(bars_per_segment):
            bar_beat_index = downbeats[bar_start_index + local_bar]
            next_beat_index = downbeats[bar_start_index + local_bar + 1]
            midpoint = float((beat_times[bar_beat_index] + beat_times[next_beat_index]) / 2.0)
            root_pc, quality_name = _parse_pop909_chord_label(
                _active_pop909_label(chords, midpoint),
                previous_root,
                previous_quality,
            )
            hold = bool(local_bar > 0 and root_pc == previous_root and quality_name == previous_quality)
            harmony_bars.append(
                HarmonyBar(
                    bar_index=local_bar,
                    root_pc=root_pc,
                    quality=quality_name,
                    hold=hold,
                    key_tonic_pc=harmony_bars[0].root_pc if harmony_bars else root_pc,
                    key_mode="major",
                )
            )
            previous_root = root_pc
            previous_quality = quality_name

        melody_events: List[MelodyEvent] = []
        seen_positions = set()
        for note_start, note_end, midi_pitch in melody_notes:
            if note_start < start_time or note_start >= end_time:
                continue
            onset_beats = _time_to_beat(note_start, beat_times) - start_beat_index
            end_beats = _time_to_beat(note_end, beat_times) - start_beat_index
            local_bar = int(np.floor(onset_beats / 4.0))
            if local_bar < 0 or local_bar >= bars_per_segment:
                continue
            position_step = int(np.clip(round((onset_beats - local_bar * 4.0) * config.steps_per_beat), 0, config.steps_per_bar - 1))
            signature = (local_bar, position_step)
            if signature in seen_positions:
                continue
            seen_positions.add(signature)
            duration_steps = int(np.clip(round((end_beats - onset_beats) * config.steps_per_beat), 1, config.steps_per_bar))
            octave = int(np.clip(midi_pitch // 12 - 1, config.melody_octave_min, config.melody_octave_max))
            melody_events.append(
                MelodyEvent(
                    bar_index=local_bar,
                    position_step=position_step,
                    duration_steps=duration_steps,
                    relative_pc=(midi_pitch - harmony_bars[local_bar].root_pc) % 12,
                    octave=octave,
                )
            )
        melody_events.sort(key=lambda item: (item.bar_index, item.position_step, item.octave))
        if len(melody_events) < config.min_notes_per_segment:
            continue
        try:
            key_signature = key.Key(PITCH_CLASS_NAMES[harmony_bars[0].root_pc])
        except Exception:
            key_signature = key.Key("C")
        descriptor_vector = _segment_descriptor_vector(harmony_bars, melody_events, key_signature, tempo_bpm, config)
        harmony_token_ids, harmony_prefix_ids = harmony_tokenizer.encode_progression(
            harmony_bars,
            descriptor_vector,
            harmony_bars[0].root_pc,
            "major",
        )
        melody_token_ids, melody_prefix_ids = melody_tokenizer.encode_melody(
            melody_events,
            harmony_bars,
            descriptor_vector,
            harmony_bars[0].root_pc,
            "major",
        )
        segments.append(
            StructuredMusicSegment(
                segment_id=f"{score_path.stem}_b{bar_start_index:03d}",
                source_path=str(score_path),
                start_measure=bar_start_index,
                end_measure=bar_start_index + bars_per_segment,
                harmony_bars=harmony_bars,
                melody_events=melody_events,
                harmony_token_ids=harmony_token_ids,
                harmony_prefix_ids=harmony_prefix_ids,
                melody_token_ids=melody_token_ids,
                melody_prefix_ids=melody_prefix_ids,
                descriptor_vector=descriptor_vector,
                tempo_bpm=tempo_bpm,
                key_tonic_pc=harmony_bars[0].root_pc,
                key_mode="major",
            )
        )
    return segments


def _segment_to_dict(segment: StructuredMusicSegment) -> dict:
    return {
        "segment_id": segment.segment_id,
        "source_path": segment.source_path,
        "start_measure": segment.start_measure,
        "end_measure": segment.end_measure,
        "harmony_bars": [asdict(item) for item in segment.harmony_bars],
        "melody_events": [asdict(item) for item in segment.melody_events],
        "harmony_token_ids": list(segment.harmony_token_ids),
        "harmony_prefix_ids": list(segment.harmony_prefix_ids),
        "melody_token_ids": list(segment.melody_token_ids),
        "melody_prefix_ids": list(segment.melody_prefix_ids),
        "descriptor_vector": [float(value) for value in segment.descriptor_vector],
        "tempo_bpm": float(segment.tempo_bpm),
        "key_tonic_pc": int(segment.key_tonic_pc),
        "key_mode": segment.key_mode,
    }


def _segment_from_dict(payload: dict) -> StructuredMusicSegment:
    return StructuredMusicSegment(
        segment_id=payload["segment_id"],
        source_path=payload["source_path"],
        start_measure=int(payload["start_measure"]),
        end_measure=int(payload["end_measure"]),
        harmony_bars=[HarmonyBar(**item) for item in payload["harmony_bars"]],
        melody_events=[MelodyEvent(**item) for item in payload["melody_events"]],
        harmony_token_ids=[int(value) for value in payload["harmony_token_ids"]],
        harmony_prefix_ids=[int(value) for value in payload["harmony_prefix_ids"]],
        melody_token_ids=[int(value) for value in payload["melody_token_ids"]],
        melody_prefix_ids=[int(value) for value in payload["melody_prefix_ids"]],
        descriptor_vector=np.asarray(payload["descriptor_vector"], dtype=np.float32),
        tempo_bpm=float(payload["tempo_bpm"]),
        key_tonic_pc=int(payload["key_tonic_pc"]),
        key_mode=payload["key_mode"],
    )


def _load_segment_cache(cache_path: Path, config: MusicDataConfig) -> Optional[List[StructuredMusicSegment]]:
    if not cache_path.exists():
        return None
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("music_config") != asdict(config):
        return None
    return [_segment_from_dict(item) for item in payload.get("segments", [])]


def _write_segment_cache(cache_path: Path, config: MusicDataConfig, segments: Sequence[StructuredMusicSegment]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_version": 1,
        "music_config": asdict(config),
        "segment_count": len(segments),
        "segments": [_segment_to_dict(segment) for segment in segments],
    }
    cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_structured_music_corpus(
    config: Optional[MusicDataConfig] = None,
    harmony_tokenizer: Optional[HarmonyTokenizer] = None,
    melody_tokenizer: Optional[MelodyTokenizer] = None,
) -> Tuple[List[StructuredMusicSegment], HarmonyTokenizer, MelodyTokenizer]:
    music_config = config or MusicDataConfig()
    _ensure_music21()
    harmony_tokenizer = harmony_tokenizer or HarmonyTokenizer(music_config)
    melody_tokenizer = melody_tokenizer or MelodyTokenizer(music_config)
    if music_config.segment_cache_path:
        cached_segments = _load_segment_cache(Path(music_config.segment_cache_path), music_config)
        if cached_segments:
            return cached_segments, harmony_tokenizer, melody_tokenizer

    midi_dirs = list(music_config.midi_dirs)
    score_files = iter_score_files(midi_dirs)
    if not score_files and music_config.use_music21_corpus_fallback:
        fallback_dir = Path(midi_dirs[0]) if midi_dirs else Path("data/midi/polyphonic_music21")
        bootstrap_music21_corpus(
            str(fallback_dir),
            max_files=music_config.max_music21_files,
            composers=music_config.music21_composers,
        )
        score_files = iter_score_files([str(fallback_dir)])
    if music_config.max_score_files > 0:
        score_files = score_files[: music_config.max_score_files]

    segments: List[StructuredMusicSegment] = []
    for score_path in score_files:
        fast_segments = _pop909_segments_from_score(score_path, harmony_tokenizer, melody_tokenizer, music_config)
        if fast_segments is None:
            fast_segments = _structured_segments_from_score(score_path, harmony_tokenizer, melody_tokenizer, music_config)
        segments.extend(fast_segments)
        if music_config.max_segments > 0 and len(segments) >= music_config.max_segments:
            segments = segments[: music_config.max_segments]
            break
    if not segments:
        raise ValueError("No valid structured chord+melody segments were found in the configured music corpus.")
    if music_config.segment_cache_path:
        _write_segment_cache(Path(music_config.segment_cache_path), music_config, segments)
    return segments, harmony_tokenizer, melody_tokenizer


def render_harmony_and_melody_to_score(
    harmony_bars: Sequence[HarmonyBar],
    melody_events: Sequence[Tuple[int, int, int, int]],
    tempo_bpm: float,
    config: MusicDataConfig,
) -> "stream.Score":
    _ensure_music21()
    score = stream.Score(id="biosonification_structured")
    harmony_part = stream.Part(id="harmony")
    melody_part = stream.Part(id="melody")
    harmony_part.partName = "Harmony"
    melody_part.partName = "Melody"
    time_signature = meter.TimeSignature("4/4")
    metronome = tempo.MetronomeMark(number=float(tempo_bpm))
    harmony_part.insert(0, time_signature)
    harmony_part.insert(0, metronome)
    melody_part.insert(0, meter.TimeSignature("4/4"))
    melody_part.insert(0, tempo.MetronomeMark(number=float(tempo_bpm)))
    total_steps = max(len(harmony_bars) * config.steps_per_bar, config.steps_per_bar)

    current_root = 0
    current_quality = "maj"
    for harmony_bar in harmony_bars:
        if not harmony_bar.hold:
            current_root = harmony_bar.root_pc
            current_quality = harmony_bar.quality
        intervals = QUALITY_INTERVALS.get(current_quality, QUALITY_INTERVALS["other"])
        root_midi = 12 * (config.chord_octave + 1) + current_root
        chord_notes = [note.Note(root_midi + interval) for interval in intervals]
        block = chord.Chord(chord_notes)
        block.quarterLength = 4.0
        block.offset = harmony_bar.bar_index * 4.0
        for pitch_note in block.notes:
            pitch_note.volume.velocity = 58
        harmony_part.insert(block.offset, block)

    for onset_step, duration_steps, midi_pitch, _ in melody_events:
        if onset_step < 0 or onset_step >= total_steps:
            continue
        duration_steps = min(int(duration_steps), total_steps - int(onset_step))
        if duration_steps <= 0:
            continue
        melodic_note = note.Note(int(midi_pitch))
        melodic_note.quarterLength = duration_steps / config.steps_per_beat
        melodic_note.offset = onset_step / config.steps_per_beat
        melodic_note.volume.velocity = 88
        melody_part.insert(melodic_note.offset, melodic_note)

    score.insert(0, harmony_part)
    score.insert(0, melody_part)
    return score
