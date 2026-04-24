from pathlib import Path

import numpy as np
import torch

from bio_music_pipeline.v2.bio import BiologicalSequenceEncoder
from bio_music_pipeline.v2.config import BioEncoderConfig, MusicDataConfig
from bio_music_pipeline.v2.dataset import NoteEvent, PolyphonicMusicTokenizer
from bio_music_pipeline.v2.model import ControlConditionedTransformer


def test_bio_encoder_quick_sample():
    encoder = BiologicalSequenceEncoder(BioEncoderConfig())
    fasta_path = Path("data/fasta/quick_sample.fa")
    results = encoder.encode_fasta(str(fasta_path))
    assert results
    assert results[0].vector.shape == (encoder.config.embedding_dim,)
    assert results[0].control_profile.shape == (6,)


def test_tokenizer_roundtrip():
    tokenizer = PolyphonicMusicTokenizer(MusicDataConfig())
    events = [
        NoteEvent(pitch=60, offset=0.0, duration=1.0, velocity=72, measure_number=1),
        NoteEvent(pitch=64, offset=0.0, duration=1.0, velocity=72, measure_number=1),
        NoteEvent(pitch=67, offset=0.0, duration=1.0, velocity=72, measure_number=1),
        NoteEvent(pitch=62, offset=1.0, duration=1.0, velocity=80, measure_number=1),
    ]
    descriptor = np.array([0.4, 0.5, 0.7, 0.5, 0.8, 1.0], dtype=np.float32)
    token_ids, control_ids = tokenizer.encode_events(events, descriptor, "major")
    assert token_ids[0] == tokenizer.bos_token_id
    assert tokenizer.sep_token_id in token_ids
    score = tokenizer.decode_to_score(token_ids)
    assert len(score.flatten().notes) >= 4


def test_model_forward_and_generate():
    tokenizer = PolyphonicMusicTokenizer(MusicDataConfig())
    model = ControlConditionedTransformer(
        vocab_size=len(tokenizer.vocab),
        bio_vector_dim=192,
        max_seq_len=128,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        sep_token_id=tokenizer.sep_token_id,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )
    input_ids = torch.randint(0, len(tokenizer.vocab), (2, 32))
    input_ids[:, -1] = tokenizer.eos_token_id
    bio_vector = torch.randn(2, 192)
    descriptor = torch.rand(2, 6)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    outputs = model.compute_loss(
        input_ids=input_ids,
        bio_vector=bio_vector,
        descriptor_vector=descriptor,
        attention_mask=attention_mask,
    )
    assert outputs["loss"].item() >= 0.0

    generated = model.generate(
        bio_vector=torch.randn(192),
        control_token_ids=tokenizer.control_tokens(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 1.0], dtype=np.float32), "major"),
        max_new_tokens=16,
        min_new_tokens=4,
    )
    assert generated.ndim == 1
    assert generated[0].item() == tokenizer.bos_token_id
