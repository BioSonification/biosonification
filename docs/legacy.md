# Legacy Components

This repository keeps the original BioSonification pipeline for historical comparison, but it is no longer the recommended path for new work.

## Current Path

Use these entrypoints for the maintained structured `v2` pipeline:

- `train_bio_music_v2.py`
- `generate_from_fasta_v2.py`
- `tools/evaluate_structured_v2.py`
- `tools/report_structured_dataset.py`
- `configs/pipeline_v2_small.json`
- `bio_music_pipeline/v2/structured_*`

The current output format is a two-track MIDI file:

- harmony: bar-level chord grid
- melody: monophonic line conditioned on generated harmony

## Legacy Entry Points

These files are retained for reproducing earlier experiments and comparing older baselines:

- `run_pipeline.py`
- `generate_from_fasta.py`
- `configs/pipeline_config.json`
- `configs/pipeline_full_paired.json`
- `configs/pipeline_quick_paired.json`
- `configs/pipeline_quick_paired_v2.json`

They use the older single-stream token generation stack and should not be cited as the active implementation unless the experiment explicitly targets the legacy system.

## Legacy Modules In `bio_music_pipeline/v2`

The following modules are still importable directly, but are no longer exported from `bio_music_pipeline.v2` as stable public APIs:

- `bio_music_pipeline/v2/dataset.py`
- `bio_music_pipeline/v2/pairing.py`
- `bio_music_pipeline/v2/model.py`
- `bio_music_pipeline/v2/train.py`
- `bio_music_pipeline/v2/generate.py`

Use the structured modules instead:

- `bio_music_pipeline/v2/structured_music.py`
- `bio_music_pipeline/v2/structured_pairing.py`
- `bio_music_pipeline/v2/structured_model.py`
- `bio_music_pipeline/v2/structured_train.py`
- `bio_music_pipeline/v2/structured_generate.py`

## Legacy Top-Level Package

Several top-level packages under `bio_music_pipeline/` also belong primarily to the original system:

- `extractors/`
- `sonification/`
- `models/`
- `baselines/`
- `evaluation/`
- `utils/`

Some ideas and helper code may still be useful, but the structured `v2` path does not depend on the legacy web generator or the old `best_model.pt` checkpoint format.

## Policy

- Do not add new features to the legacy path unless the goal is explicit comparison or reproduction.
- Do not document legacy commands as the primary quick start.
- Keep generated legacy artifacts out of git.
- Prefer structured `v2` APIs in new scripts, tests, and web code.
