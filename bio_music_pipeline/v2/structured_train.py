"""Training pipeline for the hierarchical harmony+melody generator."""

from __future__ import annotations

import json
import pickle
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .bio import BiologicalSequenceEncoder
from .config import load_v2_config
from .structured_model import BioConditionedSequenceModel
from .structured_music import (
    load_structured_music_corpus,
    render_harmony_and_melody_to_score,
)
from .structured_pairing import (
    StructuredPairedSample,
    build_structured_paired_dataset,
    save_structured_pairing_artifacts,
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_items(items: Sequence, val_fraction: float, test_fraction: float, seed: int) -> Tuple[List, List, List]:
    indices = list(range(len(items)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_test = max(1, int(len(indices) * test_fraction)) if len(indices) >= 3 else 0
    n_val = max(1, int(len(indices) * val_fraction)) if len(indices) - n_test >= 3 else 0
    test_indices = set(indices[:n_test])
    val_indices = set(indices[n_test : n_test + n_val])
    train_indices = set(indices[n_test + n_val :])
    if not train_indices:
        train_indices = set(indices)
        val_indices = set()
        test_indices = set()
    train = [items[index] for index in range(len(items)) if index in train_indices]
    val = [items[index] for index in range(len(items)) if index in val_indices]
    test = [items[index] for index in range(len(items)) if index in test_indices]
    return train, val, test


def _device_from_config(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _apply_calibration(profile: np.ndarray, calibration: Dict[str, np.ndarray]) -> np.ndarray:
    calibrated = (profile - calibration["bio_mean"]) / (calibration["bio_std"] + 1e-6)
    calibrated = calibrated * calibration["music_std"] + calibration["music_mean"]
    return np.clip(calibrated, 0.0, 1.0).astype(np.float32)


def _mode_from_profile(profile: np.ndarray) -> str:
    return "major" if float(profile[5]) >= 0.5 else "minor"


def _trusted_torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


class StructuredTokenDataset(Dataset):
    """Generic dataset for harmony or melody autoregressive training."""

    def __init__(self, records: Sequence[dict], pad_token_id: int):
        self.records = list(records)
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        token_ids = torch.tensor(record["token_ids"], dtype=torch.long)
        loss_mask = torch.tensor(record["loss_mask"], dtype=torch.bool)
        attention_mask = torch.ones_like(token_ids, dtype=torch.bool)
        return {
            "input_ids": token_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "bio_vector": torch.tensor(record["bio_vector"], dtype=torch.float32),
            "pair_weight": torch.tensor(record["pair_weight"], dtype=torch.float32),
            "sequence_id": record["sequence_id"],
            "segment_id": record["segment_id"],
        }

    def collate_fn(self, batch: Sequence[dict]) -> dict:
        max_len = max(item["input_ids"].size(0) for item in batch)
        input_ids = []
        attention_masks = []
        loss_masks = []
        for item in batch:
            pad_amount = max_len - item["input_ids"].size(0)
            if pad_amount > 0:
                padding = torch.full((pad_amount,), self.pad_token_id, dtype=torch.long)
                mask_padding = torch.zeros((pad_amount,), dtype=torch.bool)
                input_ids.append(torch.cat([item["input_ids"], padding], dim=0))
                attention_masks.append(torch.cat([item["attention_mask"], mask_padding], dim=0))
                loss_masks.append(torch.cat([item["loss_mask"], mask_padding], dim=0))
            else:
                input_ids.append(item["input_ids"])
                attention_masks.append(item["attention_mask"])
                loss_masks.append(item["loss_mask"])
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "loss_mask": torch.stack(loss_masks),
            "bio_vector": torch.stack([item["bio_vector"] for item in batch]),
            "pair_weight": torch.stack([item["pair_weight"] for item in batch]),
            "sequence_id": [item["sequence_id"] for item in batch],
            "segment_id": [item["segment_id"] for item in batch],
        }


def _build_harmony_records(samples: Sequence[StructuredPairedSample]) -> List[dict]:
    records: List[dict] = []
    for sample in samples:
        prefix_length = len(sample.harmony_prefix_ids) + 2
        loss_mask = [False] * prefix_length + [True] * (len(sample.harmony_token_ids) - prefix_length)
        records.append(
            {
                "token_ids": list(sample.harmony_token_ids),
                "loss_mask": loss_mask,
                "bio_vector": sample.bio_vector.astype(np.float32),
                "pair_weight": float(sample.pair_weight),
                "sequence_id": sample.sequence_id,
                "segment_id": sample.segment_id,
            }
        )
    return records


def _build_melody_records(samples: Sequence[StructuredPairedSample]) -> List[dict]:
    records: List[dict] = []
    for sample in samples:
        prefix_length = len(sample.melody_prefix_ids)
        loss_mask = [False] * prefix_length + [True] * (len(sample.melody_token_ids) - prefix_length)
        records.append(
            {
                "token_ids": list(sample.melody_token_ids),
                "loss_mask": loss_mask,
                "bio_vector": sample.bio_vector.astype(np.float32),
                "pair_weight": float(sample.pair_weight),
                "sequence_id": sample.sequence_id,
                "segment_id": sample.segment_id,
            }
        )
    return records


def _build_loader(
    records: Sequence[dict],
    pad_token_id: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool = False,
) -> DataLoader:
    dataset = StructuredTokenDataset(records, pad_token_id)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "collate_fn": dataset.collate_fn,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **loader_kwargs)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "loss_mask": batch["loss_mask"].to(device, non_blocking=True),
        "bio_vector": batch["bio_vector"].to(device, non_blocking=True),
        "pair_weight": batch["pair_weight"].to(device, non_blocking=True),
    }


def _evaluate_model(
    model: BioConditionedSequenceModel, loader: DataLoader, device: torch.device, amp_enabled: bool
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            device_batch = _to_device(batch, device)
            autocast_context = (
                torch.autocast(device_type="cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
            )
            with autocast_context:
                outputs = model.compute_loss(
                    input_ids=device_batch["input_ids"],
                    attention_mask=device_batch["attention_mask"],
                    loss_mask=device_batch["loss_mask"],
                    bio_vector=device_batch["bio_vector"],
                    pair_weight=device_batch["pair_weight"],
                )
            losses.append(float(outputs["loss"].item()))
    return float(np.mean(losses)) if losses else 0.0


def _train_model(
    model: BioConditionedSequenceModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    learning_rate: float,
    weight_decay: float,
    max_grad_norm: float,
    grad_accum_steps: int,
    num_epochs: int,
    patience: int,
    checkpoint_path: Path,
    label: str,
) -> List[dict]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = (
        torch.amp.GradScaler("cuda", enabled=amp_enabled)
        if device.type == "cuda"
        else torch.amp.GradScaler("cpu", enabled=False)
    )
    history: List[dict] = []
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(num_epochs):
        epoch_started_at = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_losses: List[float] = []
        for batch_index, batch in enumerate(train_loader):
            device_batch = _to_device(batch, device)
            autocast_context = (
                torch.autocast(device_type="cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
            )
            with autocast_context:
                outputs = model.compute_loss(
                    input_ids=device_batch["input_ids"],
                    attention_mask=device_batch["attention_mask"],
                    loss_mask=device_batch["loss_mask"],
                    bio_vector=device_batch["bio_vector"],
                    pair_weight=device_batch["pair_weight"],
                )
                loss = outputs["loss"] / max(grad_accum_steps, 1)
            scaler.scale(loss).backward()
            running_losses.append(float(outputs["loss"].item()))

            should_step = (batch_index + 1) % max(grad_accum_steps, 1) == 0 or (batch_index + 1) == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        train_loss = float(np.mean(running_losses)) if running_losses else 0.0
        val_loss = _evaluate_model(model, val_loader, device, amp_enabled)
        was_best = val_loss < best_val_loss
        epoch_seconds = time.time() - epoch_started_at
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_seconds": epoch_seconds,
                "best": was_best,
            }
        )
        print(
            f"[{label}] epoch {epoch + 1}/{num_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"seconds={epoch_seconds:.1f}",
            flush=True,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict()}, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    return history


def train_structured_pipeline(
    config_path: str | None = None,
    bio_cache_path: Optional[str] = None,
    music_cache_path: Optional[str] = None,
) -> Dict[str, str]:
    config = load_v2_config(config_path)
    _set_seed(config.training.seed)
    output_dir = config.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    # Load or extract biological data
    if bio_cache_path:
        print(f"Loading biological data from cache: {bio_cache_path}")
        with open(bio_cache_path, "rb") as f:
            bio_cache = pickle.load(f)
        bio_results = bio_cache["encodings"]
        print(f"Loaded {len(bio_results)} biological fragments from cache")
    else:
        print("Extracting biological data from FASTA...")
        encoder = BiologicalSequenceEncoder(config.bio)
        bio_results = encoder.encode_fasta(config.fasta_path)
        print(f"Extracted {len(bio_results)} biological fragments")

    # Load or extract music data
    if music_cache_path:
        print(f"Loading music data from cache: {music_cache_path}")
        with open(music_cache_path, "rb") as f:
            music_cache = pickle.load(f)
        structured_segments = music_cache["segments"]
        harmony_tokenizer = music_cache["harmony_tokenizer"]
        melody_tokenizer = music_cache["melody_tokenizer"]
        print(f"Loaded {len(structured_segments)} music segments from cache")
    else:
        print("Extracting music data from MIDI...")
        structured_segments, harmony_tokenizer, melody_tokenizer = load_structured_music_corpus(config.music)
        print(f"Extracted {len(structured_segments)} music segments")

    train_bio, val_bio, test_bio = _split_items(
        bio_results,
        val_fraction=config.training.val_fraction,
        test_fraction=config.training.test_fraction,
        seed=config.training.seed,
    )
    train_music, val_music, test_music = _split_items(
        structured_segments,
        val_fraction=config.training.val_fraction,
        test_fraction=config.training.test_fraction,
        seed=config.training.seed + 11,
    )
    if not val_bio:
        val_bio = train_bio[: max(1, min(4, len(train_bio)))]
    if not val_music:
        val_music = train_music[: max(1, min(16, len(train_music)))]
    if not test_bio:
        test_bio = val_bio
    if not test_music:
        test_music = val_music

    train_pairs, train_calibration = build_structured_paired_dataset(train_bio, train_music, config.pairing)
    val_pairs, _ = build_structured_paired_dataset(val_bio, val_music, config.pairing)
    test_pairs, _ = build_structured_paired_dataset(test_bio, test_music, config.pairing)
    save_structured_pairing_artifacts(str(output_dir / "pairing"), train_pairs, train_calibration)
    pin_memory = bool(
        config.training.device == "cuda" or (config.training.device == "auto" and torch.cuda.is_available())
    )

    harmony_train_loader = _build_loader(
        _build_harmony_records(train_pairs),
        harmony_tokenizer.pad_token_id,
        config.training.batch_size,
        True,
        config.training.num_workers,
        pin_memory,
    )
    harmony_val_loader = _build_loader(
        _build_harmony_records(val_pairs),
        harmony_tokenizer.pad_token_id,
        config.training.batch_size,
        False,
        config.training.num_workers,
        pin_memory,
    )
    harmony_test_loader = _build_loader(
        _build_harmony_records(test_pairs),
        harmony_tokenizer.pad_token_id,
        config.training.batch_size,
        False,
        config.training.num_workers,
        pin_memory,
    )
    melody_train_loader = _build_loader(
        _build_melody_records(train_pairs),
        melody_tokenizer.pad_token_id,
        config.training.batch_size,
        True,
        config.training.num_workers,
        pin_memory,
    )
    melody_val_loader = _build_loader(
        _build_melody_records(val_pairs),
        melody_tokenizer.pad_token_id,
        config.training.batch_size,
        False,
        config.training.num_workers,
        pin_memory,
    )
    melody_test_loader = _build_loader(
        _build_melody_records(test_pairs),
        melody_tokenizer.pad_token_id,
        config.training.batch_size,
        False,
        config.training.num_workers,
        pin_memory,
    )

    device = _device_from_config(config.training.device)
    amp_enabled = bool(device.type == "cuda" and config.training.mixed_precision)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    harmony_model = BioConditionedSequenceModel(
        vocab_size=len(harmony_tokenizer.vocab),
        bio_vector_dim=config.bio.embedding_dim,
        max_seq_len=config.training.harmony_max_seq_len,
        pad_token_id=harmony_tokenizer.pad_token_id,
        bos_token_id=harmony_tokenizer.bos_token_id,
        eos_token_id=harmony_tokenizer.eos_token_id,
        d_model=config.training.d_model,
        n_heads=config.training.n_heads,
        n_layers=config.training.n_layers,
        dim_feedforward=config.training.dim_feedforward,
        dropout=config.training.dropout,
    ).to(device)
    melody_model = BioConditionedSequenceModel(
        vocab_size=len(melody_tokenizer.vocab),
        bio_vector_dim=config.bio.embedding_dim,
        max_seq_len=config.training.melody_max_seq_len,
        pad_token_id=melody_tokenizer.pad_token_id,
        bos_token_id=melody_tokenizer.bos_token_id,
        eos_token_id=melody_tokenizer.eos_token_id,
        d_model=config.training.d_model,
        n_heads=config.training.n_heads,
        n_layers=config.training.n_layers,
        dim_feedforward=config.training.dim_feedforward,
        dropout=config.training.dropout,
    ).to(device)

    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    harmony_checkpoint = checkpoint_dir / "harmony_best.pt"
    melody_checkpoint = checkpoint_dir / "melody_best.pt"

    harmony_history = _train_model(
        harmony_model,
        harmony_train_loader,
        harmony_val_loader,
        device,
        amp_enabled,
        learning_rate=config.training.harmony_learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        grad_accum_steps=config.training.grad_accum_steps,
        num_epochs=config.training.harmony_num_epochs,
        patience=config.training.patience,
        checkpoint_path=harmony_checkpoint,
        label="harmony",
    )
    melody_history = _train_model(
        melody_model,
        melody_train_loader,
        melody_val_loader,
        device,
        amp_enabled,
        learning_rate=config.training.melody_learning_rate,
        weight_decay=config.training.weight_decay,
        max_grad_norm=config.training.max_grad_norm,
        grad_accum_steps=config.training.grad_accum_steps,
        num_epochs=config.training.melody_num_epochs,
        patience=config.training.patience,
        checkpoint_path=melody_checkpoint,
        label="melody",
    )

    harmony_state = _trusted_torch_load(harmony_checkpoint, device)
    melody_state = _trusted_torch_load(melody_checkpoint, device)
    harmony_model.load_state_dict(harmony_state["model_state_dict"])
    melody_model.load_state_dict(melody_state["model_state_dict"])
    harmony_test_loss = _evaluate_model(harmony_model, harmony_test_loader, device, amp_enabled)
    melody_test_loss = _evaluate_model(melody_model, melody_test_loader, device, amp_enabled)

    sample_bio = test_bio[0]
    sample_profile = _apply_calibration(sample_bio.control_profile, train_calibration)
    mode_name = _mode_from_profile(sample_profile)
    harmony_prefix = [
        harmony_tokenizer.bos_token_id,
        *harmony_tokenizer.control_tokens(sample_profile, sample_bio.tonic_pc_hint, mode_name),
        harmony_tokenizer.sep_token_id,
    ]
    generated_harmony = harmony_model.generate(
        bio_vector=torch.tensor(sample_bio.vector, dtype=torch.float32, device=device),
        prefix_token_ids=harmony_prefix,
        max_new_tokens=config.generation.harmony_max_new_tokens,
        temperature=config.generation.harmony_temperature,
        top_k=config.generation.harmony_top_k,
        top_p=config.generation.harmony_top_p,
        min_new_tokens=0,
        stop_token_ids=[harmony_tokenizer.eos_token_id],
    )
    harmony_bars = harmony_tokenizer.decode_progression(generated_harmony.tolist(), config.generation.num_bars)
    melody_prefix = [
        melody_tokenizer.bos_token_id,
        *melody_tokenizer.control_tokens(sample_profile, sample_bio.tonic_pc_hint, mode_name),
        *(melody_tokenizer.token_to_id[token] for token in melody_tokenizer.harmony_prefix_tokens(harmony_bars)),
        melody_tokenizer.sep_token_id,
    ]
    generated_melody = melody_model.generate(
        bio_vector=torch.tensor(sample_bio.vector, dtype=torch.float32, device=device),
        prefix_token_ids=melody_prefix,
        max_new_tokens=config.generation.melody_max_new_tokens,
        temperature=config.generation.melody_temperature,
        top_k=config.generation.melody_top_k,
        top_p=config.generation.melody_top_p,
        min_new_tokens=config.generation.melody_min_new_tokens,
        stop_token_ids=[melody_tokenizer.eos_token_id],
    )
    decoded_melody = melody_tokenizer.decode_melody(generated_melody.tolist(), harmony_bars, sample_bio.tonic_pc_hint)
    smoke_score = render_harmony_and_melody_to_score(
        harmony_bars,
        decoded_melody,
        48.0 + float(sample_profile[0]) * 120.0,
        config.music,
    )
    smoke_dir = output_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    smoke_midi_path = smoke_dir / "structured_sample.mid"
    smoke_score.write("midi", fp=str(smoke_midi_path))

    combined_checkpoint = checkpoint_dir / "structured_pipeline.pt"
    torch.save(
        {
            "harmony_model_state_dict": harmony_model.state_dict(),
            "melody_model_state_dict": melody_model.state_dict(),
            "config": asdict(config),
            "train_calibration": {key: value.astype(np.float32) for key, value in train_calibration.items()},
            "tokenizer_info": {
                "harmony_vocab_size": len(harmony_tokenizer.vocab),
                "melody_vocab_size": len(melody_tokenizer.vocab),
            },
        },
        combined_checkpoint,
    )

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "device": str(device),
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
                "gpu_total_memory_gb": (
                    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 3)
                    if torch.cuda.is_available()
                    else 0.0
                ),
                "amp_enabled": amp_enabled,
                "pin_memory": pin_memory,
                "num_workers": config.training.num_workers,
                "batch_size": config.training.batch_size,
                "grad_accum_steps": config.training.grad_accum_steps,
                "n_bio_sequences": len(bio_results),
                "n_music_segments": len(structured_segments),
                "n_train_pairs": len(train_pairs),
                "n_val_pairs": len(val_pairs),
                "n_test_pairs": len(test_pairs),
                "harmony_history": harmony_history,
                "melody_history": melody_history,
                "harmony_test_loss": harmony_test_loss,
                "melody_test_loss": melody_test_loss,
            },
            handle,
            indent=2,
        )

    return {
        "checkpoint": str(combined_checkpoint),
        "harmony_checkpoint": str(harmony_checkpoint),
        "melody_checkpoint": str(melody_checkpoint),
        "metrics": str(metrics_path),
        "sample_midi": str(smoke_midi_path),
        "output_dir": str(output_dir),
    }
