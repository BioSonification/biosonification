from pathlib import Path

import numpy as np
import torch

from bio_music_pipeline.v2.bio import BiologicalSequenceEncoder
from bio_music_pipeline.v2.config import BioEncoderConfig, MusicDataConfig
from bio_music_pipeline.v2.structured_model import BioConditionedSequenceModel
from bio_music_pipeline.v2.structured_music import (
    HarmonyBar,
    HarmonyTokenizer,
    MelodyTokenizer,
    load_structured_music_corpus,
    render_harmony_and_melody_to_score,
)


def test_bio_encoder_quick_sample():
    encoder = BiologicalSequenceEncoder(BioEncoderConfig(use_esm_embedding=False))
    fasta_path = Path("data/fasta/quick_sample.fa")
    results = encoder.encode_fasta(str(fasta_path))
    assert results
    assert results[0].vector.shape == (encoder.config.embedding_dim,)
    assert results[0].control_profile.shape == (6,)
    assert 0 <= results[0].tonic_pc_hint <= 11


def test_structured_music_extraction_and_render():
    config = MusicDataConfig(midi_dirs=["data/midi/polyphonic_music21"], max_music21_files=4)
    segments, harmony_tokenizer, melody_tokenizer = load_structured_music_corpus(config)
    assert segments
    sample = segments[0]
    assert sample.harmony_token_ids[0] == harmony_tokenizer.bos_token_id
    assert sample.melody_token_ids[0] == melody_tokenizer.bos_token_id
    decoded_harmony = harmony_tokenizer.decode_progression(sample.harmony_token_ids, config.bars_per_segment)
    decoded_melody = melody_tokenizer.decode_melody(sample.melody_token_ids, decoded_harmony, sample.key_tonic_pc)
    score = render_harmony_and_melody_to_score(decoded_harmony, decoded_melody, sample.tempo_bpm, config)
    assert len(score.parts) == 2
    assert len(score.parts[1].flatten().notes) > 0


def test_structured_model_forward_and_generate():
    music_config = MusicDataConfig()
    harmony_tokenizer = HarmonyTokenizer(music_config)
    model = BioConditionedSequenceModel(
        vocab_size=len(harmony_tokenizer.vocab),
        bio_vector_dim=256,
        max_seq_len=64,
        pad_token_id=harmony_tokenizer.pad_token_id,
        bos_token_id=harmony_tokenizer.bos_token_id,
        eos_token_id=harmony_tokenizer.eos_token_id,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    )
    input_ids = torch.randint(0, len(harmony_tokenizer.vocab), (2, 16))
    input_ids[:, -1] = harmony_tokenizer.eos_token_id
    bio_vector = torch.randn(2, 256)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    loss_mask = torch.ones_like(input_ids, dtype=torch.bool)
    loss_mask[:, :4] = False
    outputs = model.compute_loss(
        input_ids=input_ids,
        bio_vector=bio_vector,
        attention_mask=attention_mask,
        loss_mask=loss_mask,
    )
    assert outputs["loss"].item() >= 0.0

    prefix = [
        harmony_tokenizer.bos_token_id,
        *harmony_tokenizer.control_tokens(np.array([0.5, 0.5, 0.5, 0.5, 0.5, 1.0], dtype=np.float32), 0, "major"),
        harmony_tokenizer.sep_token_id,
    ]
    generated = model.generate(
        bio_vector=torch.randn(256),
        prefix_token_ids=prefix,
        max_new_tokens=8,
        temperature=0.9,
        top_k=8,
        top_p=0.9,
        stop_token_ids=[harmony_tokenizer.eos_token_id],
    )
    assert generated.ndim == 1
    assert generated[0].item() == harmony_tokenizer.bos_token_id


def test_melody_decode_is_bounded_and_monophonic():
    config = MusicDataConfig(bars_per_segment=2)
    tokenizer = MelodyTokenizer(config)
    harmony_bars = [
        HarmonyBar(bar_index=0, root_pc=0, quality="maj", hold=False, key_tonic_pc=0, key_mode="major"),
        HarmonyBar(bar_index=1, root_pc=7, quality="maj", hold=False, key_tonic_pc=0, key_mode="major"),
    ]
    token_ids = [
        tokenizer.bos_token_id,
        tokenizer.sep_token_id,
        tokenizer.token_to_id["BAR"],
        tokenizer.token_to_id["POS_0"],
        tokenizer.token_to_id["RELPC_0"],
        tokenizer.token_to_id["OCT_5"],
        tokenizer.token_to_id["DUR_8"],
        tokenizer.token_to_id["POS_0"],
        tokenizer.token_to_id["RELPC_4"],
        tokenizer.token_to_id["OCT_6"],
        tokenizer.token_to_id["DUR_4"],
        tokenizer.token_to_id["POS_2"],
        tokenizer.token_to_id["RELPC_7"],
        tokenizer.token_to_id["OCT_5"],
        tokenizer.token_to_id["DUR_16"],
        tokenizer.token_to_id["BAR"],
        tokenizer.token_to_id["POS_1"],
        tokenizer.token_to_id["RELPC_0"],
        tokenizer.token_to_id["OCT_5"],
        tokenizer.token_to_id["DUR_16"],
        tokenizer.token_to_id["BAR"],
        tokenizer.token_to_id["POS_0"],
        tokenizer.token_to_id["RELPC_0"],
        tokenizer.token_to_id["OCT_5"],
        tokenizer.token_to_id["DUR_2"],
        tokenizer.eos_token_id,
    ]
    events = tokenizer.decode_melody(token_ids, harmony_bars, tonic_pc=0)
    assert events
    total_steps = len(harmony_bars) * config.steps_per_bar
    assert max(onset + duration for onset, duration, _, _ in events) <= total_steps
    for index in range(len(events) - 1):
        current_onset, current_duration, _, _ = events[index]
        next_onset, _, _, _ = events[index + 1]
        assert next_onset >= current_onset + current_duration
