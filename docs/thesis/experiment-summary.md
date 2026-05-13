# Thesis Experiment Summary

This document records the final local experiment prepared for the diploma text.

## Final Run

| Item | Value |
|---|---|
| Config | `configs/pipeline_v2_thesis_rtx2060.json` |
| Output directory | `results/thesis_final_run_regularized` |
| Checkpoint | `results/thesis_final_run_regularized/checkpoints/structured_pipeline.pt` |
| Dataset report | `results/thesis_dataset_report/dataset_report.md` |
| Evaluation report | `results/thesis_evaluation/evaluation_report.md` |
| Training plot | `results/thesis_artifacts/training_loss_curves.png` |
| Evaluation plot | `results/thesis_artifacts/evaluation_metric_comparison.png` |

## Hardware And Training Mode

| Item | Value |
|---|---|
| GPU | `NVIDIA GeForce RTX 2060` |
| VRAM | `6 GB` |
| Device used by training | `cuda` |
| Mixed precision | enabled |
| Batch size | `4` |
| Gradient accumulation | `4` |
| Effective batch size | `16` |

## Data

| Source | Use |
|---|---|
| POP909 | Main music corpus for final training |
| MAESTRO v3.0.0 MIDI-only | Downloaded and inspected, but not used for the final run because `music21` preprocessing was too slow for this local workflow |
| NCBI RefSeq `GCF_000005845.2` | `Escherichia coli` K-12 MG1655 FASTA source |
| NCBI RefSeq `GCF_000146045.2` | `Saccharomyces cerevisiae` S288C FASTA source |

Resolved training data:

- FASTA records: `18`
- encoded bio fragments: `432`
- MIDI files found under POP909: `2898`
- structured music segments used: `2000`
- train pairs: `1625`
- validation pairs: `320`
- test pairs: `215`

`ViennaRNA` was disabled in the final config because folding 1800-base fragments dominated preprocessing time on CPU. The final run keeps nucleotide, k-mer and translated protein features, and keeps `ESM` disabled to fit the RTX 2060 workflow.

## Training Results

| Metric | Value |
|---|---:|
| Best harmony validation loss | `0.2112` |
| Harmony test loss | `0.2187` |
| Best melody validation loss | `0.2556` |
| Melody test loss | `0.2561` |

A first training pass overfit earlier. The final run increased regularization and reduced learning rate, improving harmony validation loss from `0.2692` to `0.2112` and melody validation loss from `0.3441` to `0.2556`.

## Evaluation Results

Evaluation used `12` generated FASTA fragments and a random harmony+melody baseline.

| Metric | Structured v2 | Random baseline |
|---|---:|---:|
| Valid MIDI rate | `12/12` | `12/12` |
| Chord-tone ratio | `0.6045` | `0.4043` |
| Melody notes | `43.50` | `42.33` |
| Note density per bar | `5.4375` | `5.2917` |
| Pitch range | `14.58` | `34.17` |
| Unique pitches | `10.75` | `22.83` |
| Self-similarity | `0.1382` | `0.1086` |

Interpretation: the model produces valid two-part MIDI and places melody notes inside generated harmony more often than the random baseline. These metrics do not prove a causal biological-musical relationship; they support the narrower claim that biological sequence features can condition a structured symbolic music generator.

## Dataset References

- POP909 dataset repository: https://github.com/music-x-lab/POP909-Dataset
- MAESTRO dataset page: https://magenta.tensorflow.org/datasets/maestro
- NCBI dataset page for `GCF_000005845.2`: https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000005845.2/
- NCBI dataset page for `GCF_000146045.2`: https://www.ncbi.nlm.nih.gov/datasets/genome/GCF_000146045.2/
