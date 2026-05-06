# Dataset Summary for BioSonification Project

**Generated:** 2026-05-05  
**Status:** ✅ Ready for large-scale training

---

## 📊 Overview

| Category | Count | Size |
|----------|-------|------|
| **Biological sequences** | 5 genomes | 372 MB |
| **Estimated bio fragments** | ~433,473 | - |
| **MIDI files** | 4,174 | ~2.9 GB |
| **Estimated music segments** | ~15,000-20,000 | - |

---

## 🧬 Biological Data

### Downloaded Reference Genomes

All genomes downloaded from NCBI RefSeq:

| Organism | Accession | Size (MB) | Fragments* | Description |
|----------|-----------|-----------|------------|-------------|
| **E. coli K-12** | GCF_000005845.2 | 4.8 | ~5,285 | Model bacterium |
| **S. cerevisiae** | GCF_000146045.2 | 12.5 | ~13,846 | Baker's yeast |
| **C. elegans** | GCF_000002985.6 | 102.8 | ~114,214 | Nematode worm |
| **A. thaliana** | GCF_000001735.4 | 122.7 | ~136,288 | Model plant |
| **D. melanogaster** | GCF_000001215.4 | 147.5 | ~163,840 | Fruit fly |
| **TOTAL** | - | **372.1** | **~433,473** | - |

*Fragments calculated with `fragment_length=1800`, `stride=900`

### Location
```
data/fasta/refseq_genomes/
├── GCF_000005845.2_genomic.fna  (E. coli)
├── GCF_000146045.2_genomic.fna  (S. cerevisiae)
├── GCF_000002985.6_genomic.fna  (C. elegans)
├── GCF_000001735.4_genomic.fna  (A. thaliana)
└── GCF_000001215.4_genomic.fna  (D. melanogaster)
```

---

## 🎵 Musical Data

### MIDI Corpora

| Corpus | Files | Description |
|--------|-------|-------------|
| **POP909** | 1,276 | Chinese pop songs with chord annotations |
| **maestro-v3.0.0** | 2,898 | Classical piano performances (MIDI from audio) |
| **TOTAL** | **4,174** | - |

### Estimated Music Segments

With current config (`bars_per_segment=4`, `segment_hop_bars=2`):
- **POP909:** ~5,000-7,000 segments
- **maestro:** ~10,000-13,000 segments
- **TOTAL:** ~15,000-20,000 segments

### Location
```
data/midi/
├── POP909/POP909/           (1,276 MIDI files)
└── maestro-v3.0.0/          (2,898 MIDI files)
```

---

## 📈 Dataset Comparison

### Before (pipeline_v2_small.json)
- Bio sequences: 12
- Bio fragments: ~30
- Music segments: 265
- Train pairs: 30

### After (pipeline_v2_large.json)
- Bio sequences: 5 genomes
- Bio fragments: **~433,473** (14,449x increase)
- Music segments: **~15,000-20,000** (60x increase)
- Expected train pairs: **~50,000-100,000** (1,667x-3,333x increase)

---

## 🚀 Next Steps

### 1. Generate Dataset Report
```powershell
.\.venv\Scripts\python.exe tools\report_structured_dataset.py `
  --config configs\pipeline_v2_large.json `
  --output-dir results\v2_large_dataset_report
```

This will:
- Count actual music segments extracted
- Build bio-music pairing manifest
- Estimate training time

### 2. Train Large Model
```powershell
.\.venv\Scripts\python.exe train_bio_music_v2.py `
  --config configs\pipeline_v2_large.json
```

Expected improvements:
- **Pitch range:** 14.6 → 25+ (wider melodies)
- **Unique pitches:** 10.8 → 18+ (more variety)
- **Pitch entropy:** 2.86 → 3.2+ (less repetitive)
- **Chord-tone ratio:** 60% → 70%+ (better harmony following)

### 3. Evaluation
```powershell
.\.venv\Scripts\python.exe tools\evaluate_structured_v2.py `
  --checkpoint results\v2_large_dataset\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output-dir results\v2_large_evaluation `
  --max-records 20
```

---

## 💾 Storage Requirements

| Component | Size |
|-----------|------|
| Raw genomes | 372 MB |
| MIDI files | ~2.9 GB |
| Processed features (estimated) | ~5 GB |
| Model checkpoints | ~2 GB |
| **TOTAL** | **~10 GB** |

---

## 📝 Configuration Files

- **Small dataset (RTX 2060):** `configs/pipeline_v2_small.json`
- **Large dataset (A100/H100):** `configs/pipeline_v2_large.json`

### Key Differences

| Parameter | Small | Large |
|-----------|-------|-------|
| `fasta_path` | `quick_sample.fa` | `refseq_genomes/` |
| `midi_dirs` | `polyphonic_music21` | `POP909 + maestro` |
| `use_esm_embedding` | `false` | `true` |
| `d_model` | 256 | 512 |
| `n_layers` | 4 | 8 |
| `batch_size` | 4 | 32 |
| `num_epochs` | 10 | 50 |

---

## 🔬 Scientific Value

### Phylogenetic Diversity
- **Bacteria:** E. coli
- **Fungi:** S. cerevisiae
- **Animals:** C. elegans, D. melanogaster
- **Plants:** A. thaliana

This covers major domains of life, allowing the model to learn diverse biological patterns.

### Musical Diversity
- **POP909:** Modern pop, chord progressions, vocal melodies
- **maestro:** Classical piano, complex harmonies, virtuosic passages

This provides both simple and complex musical structures.

---

## ⚠️ Important Notes

1. **ESM embeddings enabled** in large config — requires ~8GB GPU memory
2. **Training time:** Expect 24-48 hours on A100 for 50 epochs
3. **Data augmentation:** Consider adding transpose/tempo variations
4. **Ablation study:** Train 3 variants (no_bio, random_bio, real_bio) to validate bio conditioning

---

## 📚 References

- **NCBI RefSeq:** https://ftp.ncbi.nlm.nih.gov/genomes/refseq/
- **POP909:** https://github.com/music-x-lab/POP909-Dataset
- **maestro:** https://magenta.tensorflow.org/datasets/maestro
- **ESM:** https://github.com/facebookresearch/esm
