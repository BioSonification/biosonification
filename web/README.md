# BioSonification Web Interface

Flask interface for the structured `v2` biosonification pipeline with **fragmented generation**. It generates a two-track symbolic MIDI file from FASTA input:

- harmony track: bar-level chord grid
- melody track: monophonic melody conditioned on the generated harmony

## Key Features

- **Fragmented Generation**: Automatically splits long sequences into fragments and generates high-quality music for each
- **4-Bar Model**: Uses the best-performing model (val_loss 0.145 for harmony, 0.157 for melody)
- **Adaptive Length**: Longer sequences → longer compositions without quality degradation
- **FASTA Input**: Paste sequence or upload file (drag & drop supported)
- **Examples Gallery**: Pre-generated compositions from different organisms
- **Audio Playback**: Listen to examples directly in browser (requires fluidsynth/timidity)
- **MIDI Download**: Download generated and example MIDI files
- **Metadata Display**: View generation parameters (tempo, key, bars, notes, fragments)

## Requirements

Install the project dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The web interface automatically uses:
- **Model**: `results/v2_medium_rtx2060_fast/checkpoints/structured_pipeline.pt` (4-bar model, best quality)
- **Config**: `configs/pipeline_v2_medium_rtx2060_fast.json`
- **Generation**: Fragmented approach with `bars_per_fragment=4`

Optional overrides:

```bash
export BIOSONIFICATION_STRUCTURED_CHECKPOINT=/absolute/path/to/structured_pipeline.pt
export BIOSONIFICATION_CONFIG_PATH=/absolute/path/to/config.json
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

**Fragmented Generation:**
- Sequences are automatically split into 1800 bp fragments
- Each fragment generates 4 bars of music
- All fragments are concatenated into one MIDI file

**Examples:**
- 1800 bp → 1 fragment → 4 bars (~8 seconds)
- 3600 bp → 2 fragments → 8 bars (~16 seconds)
- 10000 bp → 6 fragments → 24 bars (~48 seconds)

## API

- `GET /api/status` returns generator readiness, structured checkpoint path, config path, and audio synthesizer status.
- `POST /api/generate` accepts JSON `{ "fasta": "..." }` or multipart `fasta_file`.
- `GET /api/download/<session_id>/midi` downloads the generated MIDI.
- `GET /api/download/<session_id>/wav` downloads rendered WAV when an optional synthesizer is available.
- `GET /api/examples` returns list of example compositions with metadata.
- `GET /api/examples/<example_id>/midi` downloads example MIDI file.
- `GET /api/examples/<example_id>/audio` streams example audio (WAV, converts on first request).

The generation response includes structured metadata:

- `sequence_id`
- `sequence_type`
- `full_sequence_length`
- `num_fragments`
- `bars_per_fragment`
- `total_bars`
- `total_melody_notes`
- `fragments` — array with per-fragment details:
  - `fragment_index`
  - `start_position`
  - `fragment_length`
  - `tempo_bpm`
  - `harmony_bars`
  - `melody_notes`

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

## Technical Implementation

**Generator Backend** (`web/generator.py`):
- Uses `generate_structured_music_from_fasta_fragmented()` for all generations
- Automatically selects the 4-bar model (`v2_medium_rtx2060_fast`) for best quality
- Fragments long sequences (>1800 bp) into multiple segments
- Concatenates all segments into a single MIDI file

**Model Selection Priority**:
1. `results/v2_medium_rtx2060_fast/checkpoints/structured_pipeline.pt` (4-bar, best)
2. `results/v2_medium_rtx2060_long/checkpoints/structured_pipeline.pt` (8-bar, fallback)
3. Newest checkpoint in `results/*/checkpoints/structured_pipeline.pt`

**Why Fragmented Generation**:
- Models trained on short segments (4-8 bars) produce poor quality for long compositions
- Fragmentation keeps each segment within the training distribution
- Result: stable quality regardless of sequence length
