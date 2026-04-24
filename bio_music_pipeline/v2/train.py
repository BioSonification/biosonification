"""Training entrypoints for the v2 biosonification pipeline."""

from __future__ import annotations

from dataclasses import asdict
from contextlib import nullcontext
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from .bio import BioEncodingResult, BiologicalSequenceEncoder
from .config import V2PipelineConfig, load_v2_config
from .dataset import BioMusicPairDataset, PolyphonicMusicTokenizer, load_music_corpus
from .model import ControlConditionedTransformer
from .pairing import PairedSample, build_paired_dataset, save_pairing_artifacts


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
    val_indices = set(indices[n_test:n_test + n_val])
    train_indices = set(indices[n_test + n_val:])
    if not train_indices:
        train_indices = set(indices)
        val_indices = set()
        test_indices = set()
    train = [items[index] for index in range(len(items)) if index in train_indices]
    val = [items[index] for index in range(len(items)) if index in val_indices]
    test = [items[index] for index in range(len(items)) if index in test_indices]
    return train, val, test


def _paired_samples_to_records(samples: Sequence[PairedSample]) -> List[dict]:
    return [
        {
            "sequence_id": sample.sequence_id,
            "segment_id": sample.segment_id,
            "bio_vector": sample.bio_vector.astype(np.float32),
            "descriptor_vector": sample.descriptor_vector.astype(np.float32),
            "token_ids": list(sample.token_ids),
            "pair_weight": float(sample.pair_weight),
        }
        for sample in samples
    ]


def _device_from_config(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def _mode_from_profile(profile: np.ndarray) -> str:
    return "major" if float(profile[5]) >= 0.5 else "minor"


def _apply_calibration(profile: np.ndarray, calibration: Dict[str, np.ndarray]) -> np.ndarray:
    calibrated = ((profile - calibration["bio_mean"]) / (calibration["bio_std"] + 1e-6))
    calibrated = calibrated * calibration["music_std"] + calibration["music_mean"]
    return np.clip(calibrated, 0.0, 1.0).astype(np.float32)


def _trusted_torch_load(path: Path, device: torch.device) -> dict:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _evaluate(
    model: ControlConditionedTransformer,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    losses = []
    token_losses = []
    descriptor_losses = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bio_vector = batch["bio_vector"].to(device)
            descriptor_vector = batch["descriptor_vector"].to(device)
            pair_weight = batch["pair_weight"].to(device)
            autocast_context = torch.autocast(device_type="cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
            with autocast_context:
                outputs = model.compute_loss(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bio_vector=bio_vector,
                    descriptor_vector=descriptor_vector,
                    pair_weight=pair_weight,
                )
            losses.append(float(outputs["loss"].item()))
            token_losses.append(float(outputs["token_loss"].item()))
            descriptor_losses.append(float(outputs["descriptor_loss"].item()))
    if not losses:
        return {"loss": 0.0, "token_loss": 0.0, "descriptor_loss": 0.0}
    return {
        "loss": float(np.mean(losses)),
        "token_loss": float(np.mean(token_losses)),
        "descriptor_loss": float(np.mean(descriptor_losses)),
    }


def _build_loaders(
    train_samples: Sequence[PairedSample],
    val_samples: Sequence[PairedSample],
    test_samples: Sequence[PairedSample],
    tokenizer: PolyphonicMusicTokenizer,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = BioMusicPairDataset(_paired_samples_to_records(train_samples), tokenizer.pad_token_id)
    val_dataset = BioMusicPairDataset(_paired_samples_to_records(val_samples), tokenizer.pad_token_id)
    test_dataset = BioMusicPairDataset(_paired_samples_to_records(test_samples), tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=val_dataset.collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=test_dataset.collate_fn,
    )
    return train_loader, val_loader, test_loader


def train_pipeline(config_path: str | None = None) -> Dict[str, str]:
    """Run the full v2 training pipeline."""

    config = load_v2_config(config_path)
    _set_seed(config.training.seed)
    output_dir = config.output_path
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "resolved_config.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    encoder = BiologicalSequenceEncoder(config.bio)
    bio_results = encoder.encode_fasta(config.fasta_path)
    music_segments, tokenizer = load_music_corpus(config.music)

    train_bio, val_bio, test_bio = _split_items(
        bio_results,
        val_fraction=config.training.val_fraction,
        test_fraction=config.training.test_fraction,
        seed=config.training.seed,
    )
    train_music, val_music, test_music = _split_items(
        music_segments,
        val_fraction=config.training.val_fraction,
        test_fraction=config.training.test_fraction,
        seed=config.training.seed + 7,
    )
    if not val_bio:
        val_bio = train_bio[: max(1, min(4, len(train_bio)))]
    if not val_music:
        val_music = train_music[: max(1, min(16, len(train_music)))]
    if not test_bio:
        test_bio = val_bio
    if not test_music:
        test_music = val_music

    train_pairs, train_calibration = build_paired_dataset(train_bio, train_music, config.pairing)
    val_pairs, _ = build_paired_dataset(val_bio, val_music, config.pairing)
    test_pairs, _ = build_paired_dataset(test_bio, test_music, config.pairing)
    save_pairing_artifacts(str(output_dir / "pairing"), train_pairs, train_calibration)

    train_loader, val_loader, test_loader = _build_loaders(
        train_pairs,
        val_pairs,
        test_pairs,
        tokenizer,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers,
    )

    device = _device_from_config(config.training.device)
    amp_enabled = bool(device.type == "cuda" and config.training.mixed_precision)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    model = ControlConditionedTransformer(
        vocab_size=len(tokenizer.vocab),
        bio_vector_dim=config.bio.embedding_dim,
        max_seq_len=config.training.max_seq_len,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        d_model=config.training.d_model,
        n_heads=config.training.n_heads,
        n_layers=config.training.n_layers,
        dim_feedforward=config.training.dim_feedforward,
        dropout=config.training.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else torch.amp.GradScaler("cpu", enabled=False)

    best_val_loss = float("inf")
    patience_counter = 0
    history = []
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best_model.pt"

    for epoch in range(config.training.num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = []
        for batch_index, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            bio_vector = batch["bio_vector"].to(device)
            descriptor_vector = batch["descriptor_vector"].to(device)
            pair_weight = batch["pair_weight"].to(device)
            autocast_context = torch.autocast(device_type="cuda", enabled=amp_enabled) if device.type == "cuda" else nullcontext()
            with autocast_context:
                outputs = model.compute_loss(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    bio_vector=bio_vector,
                    descriptor_vector=descriptor_vector,
                    pair_weight=pair_weight,
                )
                loss = outputs["loss"] / max(config.training.grad_accum_steps, 1)
            scaler.scale(loss).backward()
            running_loss.append(float(outputs["loss"].item()))

            should_step = (batch_index + 1) % max(config.training.grad_accum_steps, 1) == 0
            if should_step or (batch_index + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        train_metrics = {
            "loss": float(np.mean(running_loss)) if running_loss else 0.0,
        }
        val_metrics = _evaluate(model, val_loader, device, amp_enabled)
        history.append({"epoch": epoch + 1, "train": train_metrics, "val": val_metrics})

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(config),
                    "tokenizer_vocab": tokenizer.vocab,
                    "train_calibration": {key: value.astype(np.float32) for key, value in train_calibration.items()},
                },
                best_checkpoint,
            )
        else:
            patience_counter += 1
            if patience_counter >= config.training.patience:
                break

    checkpoint = _trusted_torch_load(best_checkpoint, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _evaluate(model, test_loader, device, amp_enabled)

    sample_bio = test_bio[0]
    sample_profile = _apply_calibration(sample_bio.control_profile, train_calibration)
    control_tokens = tokenizer.control_tokens(sample_profile, _mode_from_profile(sample_profile))
    generated = model.generate(
        bio_vector=torch.tensor(sample_bio.vector, dtype=torch.float32, device=device),
        control_token_ids=control_tokens,
        max_new_tokens=config.generation.max_new_tokens,
        temperature=config.generation.temperature,
        top_k=config.generation.top_k,
        top_p=config.generation.top_p,
        min_new_tokens=config.generation.min_new_tokens,
    )
    smoke_dir = output_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    sample_midi_path = smoke_dir / "sample_from_training_pipeline.mid"
    tokenizer.write_midi(
        generated.tolist(),
        str(sample_midi_path),
        tempo_bpm=48.0 + float(sample_profile[0]) * 120.0,
    )

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "history": history,
                "test_metrics": test_metrics,
                "device": str(device),
                "n_bio_sequences": len(bio_results),
                "n_music_segments": len(music_segments),
                "n_train_pairs": len(train_pairs),
                "n_val_pairs": len(val_pairs),
                "n_test_pairs": len(test_pairs),
            },
            handle,
            indent=2,
        )

    return {
        "checkpoint": str(best_checkpoint),
        "metrics": str(metrics_path),
        "sample_midi": str(sample_midi_path),
        "output_dir": str(output_dir),
    }
