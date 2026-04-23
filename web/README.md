# BioSonification Web Interface

Web application for generating music from biological FASTA sequences.

## Overview

This web interface wraps the bio-music pipeline to provide an easy-to-use interface for generating unique music from DNA sequences. The trained model is loaded once and reused for multiple generations — no need to retrain the pipeline.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prerequisites

You must have a trained model before using the web interface. If you haven't trained one yet:

```bash
python run_pipeline.py --config configs/pipeline_full_paired.json --midi-dir data/midi --paired-data results/paired_data
```

By default the web app auto-detects:

1. `results/full_paired_run/models/best_model.pt`
2. `results/models/best_model.pt`
3. newest `results/*/models/best_model.pt`

You can force a specific checkpoint via:

```bash
export BIOSONIFICATION_MODEL_PATH=/absolute/path/to/best_model.pt
```

### 3. Run the Web App

```bash
python -m web.app
```

Open **http://localhost:5001** in your browser.

Optional runtime settings:

```bash
export BIOSONIFICATION_HOST=127.0.0.1
export BIOSONIFICATION_PORT=5001
export BIOSONIFICATION_DEBUG=0
```

## Features

- **Paste or Upload**: Paste DNA sequence directly or upload FASTA file
- **Music Generation**: Transforms DNA sequences into unique MIDI music
- **Audio Playback**: Currently disabled in this build (MIDI download is available)
- **MIDI Download**: Download MIDI files for use in any DAW or player
- **Musical Parameters**: View tempo, key, scale, and other parameters derived from your DNA

## Optional: Audio Playback

In-browser WAV playback is currently disabled due to synthesizer CLI compatibility.
You can still download MIDI files and play them in DAWs or MIDI players.

## Usage

1. Open http://localhost:5001
2. Paste a DNA sequence (min 100 nucleotides) or upload a FASTA file
3. Click **"Generate Music"**
4. Wait for generation (usually takes a few seconds)
5. Download the MIDI file
6. View the musical parameters derived from your DNA

## API Endpoints

- `POST /api/generate` — Generate music from FASTA
  - Body: JSON with `fasta` field OR multipart form with `fasta_file`
  - Returns: JSON with session_id, musical_params, audio availability
  
- `GET /api/download/<session_id>/midi` — Download MIDI file
- `GET /api/download/<session_id>/wav` — Download WAV audio
- `GET /api/status` — Check application status

## File Structure

```
web/
├── app.py                 # Flask application
├── generator.py           # Music generation logic
├── midi_to_audio.py       # MIDI to WAV conversion
├── templates/
│   └── index.html         # Main page
├── static/
│   ├── css/
│   │   └── style.css      # Styles
│   └── js/
│       └── app.js         # Frontend logic
└── output/                # Generated files (auto-created)
    ├── midi/
    └── audio/
```

## Keyboard Shortcuts

- `Ctrl/Cmd + Enter` — Generate music (when on input page)

## Troubleshooting

### "Trained model not found"
Run the full pipeline first, or set:

```bash
export BIOSONIFICATION_MODEL_PATH=/absolute/path/to/best_model.pt
```

### No audio playback
Install fluidsynth or timidity (see "Optional: Audio Playback" above)

### "Sequence too short"
Provide at least 100 nucleotides (A, C, G, T only)

### Generation is slow
Long sequences (>10,000 nucleotides) take more time to process. Try shorter sequences.

## Disclaimer

This system uses bio-vectors as structured conditioning signals for music generation. **No causal relationship between genes and music is claimed or demonstrated.** Bio-vectors serve as deterministic control signals that statistically influence musical structure.
