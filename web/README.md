# BioSonification Web Interface

Flask interface for the current structured `v2` biosonification pipeline. It generates a two-track symbolic MIDI file from FASTA input:

- harmony track: bar-level chord grid
- melody track: monophonic melody conditioned on the generated harmony

The legacy `run_pipeline.py` / `best_model.pt` web path is no longer the recommended route.

## Requirements

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

Train the structured `v2` model first, or point the web app at an existing checkpoint:

```bash
python train_bio_music_v2.py --config configs/pipeline_v2_small.json
```

By default the web app looks for:

1. `results/v2_music21_rtx2060/checkpoints/structured_pipeline.pt`
2. newest `results/*/checkpoints/structured_pipeline.pt`

Optional overrides:

```bash
export BIOSONIFICATION_STRUCTURED_CHECKPOINT=/absolute/path/to/structured_pipeline.pt
export BIOSONIFICATION_CONFIG_PATH=/absolute/path/to/pipeline_v2_small.json
export BIOSONIFICATION_DEVICE=auto
```

`BIOSONIFICATION_DEVICE` accepts `auto`, `cpu`, or `cuda`.

## Run

```bash
python -m web.app
```

Open `http://localhost:5001`.

Optional server settings:

```bash
export BIOSONIFICATION_HOST=127.0.0.1
export BIOSONIFICATION_PORT=5001
export BIOSONIFICATION_DEBUG=0
```

## Input

The web form accepts pasted text or FASTA upload. DNA, RNA, and protein-like sequences are accepted by the structured `v2` encoder. The minimum cleaned sequence length is 90 symbols for the default config.

## API

- `GET /api/status` returns generator readiness, structured checkpoint path, config path, and audio synthesizer status.
- `POST /api/generate` accepts JSON `{ "fasta": "..." }` or multipart `fasta_file`.
- `GET /api/download/<session_id>/midi` downloads the generated MIDI.
- `GET /api/download/<session_id>/wav` downloads rendered WAV when an optional synthesizer is available.

The generation response includes structured metadata:

- `sequence_id`
- `sequence_type`
- `cleaned_sequence_length`
- `tempo_bpm`
- `tonic_pc_hint`
- `generated_harmony_bars`
- `generated_melody_note_count`

## Output

Runtime files are written under `web/output/`:

```text
web/output/
├── fasta/
├── metadata/
├── midi/
└── audio/
```

This directory is ignored by git.

## Scientific Note

The system uses biological features as structured conditioning signals for symbolic music generation. It does not demonstrate or claim a causal relationship between genes and music.
