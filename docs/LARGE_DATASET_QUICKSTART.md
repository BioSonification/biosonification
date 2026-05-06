# Quick Start Guide: Large Dataset Training

This guide shows how to use the newly downloaded large dataset for training.

## 📊 What's New

- **433,473 biological fragments** (from 5 reference genomes)
- **~15,000-20,000 music segments** (from POP909 + maestro)
- **Expected 50,000-100,000 training pairs** (vs 30 before)

See [DATA_SUMMARY.md](../DATA_SUMMARY.md) for full details.

---

## 🚀 Step 1: Verify Dataset

Check that all data is ready:

```powershell
# Count genome files
Get-ChildItem data\fasta\refseq_genomes\*.fna | Measure-Object

# Count MIDI files
Get-ChildItem data\midi -Recurse -Filter *.mid | Measure-Object

# Generate dataset report
.\.venv\Scripts\python.exe tools\report_structured_dataset.py `
  --config configs\pipeline_v2_large.json `
  --output-dir results\v2_large_dataset_report
```

Expected output:
- 5 genome files
- ~4,174 MIDI files
- Dataset report showing actual segment counts

---

## 🎓 Step 2: Train the Model

### Option A: Full Training (Recommended)

Train on the complete large dataset:

```powershell
.\.venv\Scripts\python.exe train_bio_music_v2.py `
  --config configs\pipeline_v2_large.json
```

**Requirements:**
- GPU: A100 (40GB) or H100
- Time: 24-48 hours for 50 epochs
- Disk: ~15 GB free space

**What happens:**
1. Loads 5 genomes → extracts ~433k bio fragments
2. Loads 4,174 MIDI files → extracts ~15k-20k music segments
3. Builds bio-music pairs with structured pairing
4. Trains harmony model (50 epochs)
5. Trains melody model (60 epochs)
6. Saves checkpoints to `results/v2_large_dataset/checkpoints/`

### Option B: Quick Test (5% of data)

Test the pipeline on a subset first:

```powershell
# Edit configs/pipeline_v2_large.json temporarily:
# - Set "max_fragments_per_record": 10 (instead of 100)
# - Set "num_epochs": 5 (instead of 50)

.\.venv\Scripts\python.exe train_bio_music_v2.py `
  --config configs\pipeline_v2_large.json
```

This will use ~5,000 bio fragments and finish in 2-3 hours.

---

## 📈 Step 3: Monitor Training

Watch the training progress:

```powershell
# View metrics in real-time
Get-Content results\v2_large_dataset\metrics.json -Wait

# Check GPU usage
nvidia-smi -l 1
```

**Good signs:**
- `harmony_loss` decreasing from ~1.5 to <0.4
- `melody_loss` decreasing from ~1.8 to <0.5
- No NaN or Inf values
- GPU utilization >80%

**Bad signs:**
- Loss not decreasing after 5 epochs → learning rate too high/low
- GPU utilization <50% → increase batch_size or num_workers
- Out of memory → decrease batch_size or d_model

---

## 🎵 Step 4: Generate Music

After training completes:

```powershell
# Generate from E. coli genome
.\.venv\Scripts\python.exe generate_from_fasta_v2.py `
  --config configs\pipeline_v2_large.json `
  --checkpoint results\v2_large_dataset\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output results\v2_large_generation\ecoli_music.mid `
  --metadata-output results\v2_large_generation\ecoli_music.json

# Generate from yeast genome
.\.venv\Scripts\python.exe generate_from_fasta_v2.py `
  --config configs\pipeline_v2_large.json `
  --checkpoint results\v2_large_dataset\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000146045.2_genomic.fna `
  --output results\v2_large_generation\yeast_music.mid `
  --metadata-output results\v2_large_generation\yeast_music.json
```

---

## 📊 Step 5: Evaluate Results

Run comprehensive evaluation:

```powershell
.\.venv\Scripts\python.exe tools\evaluate_structured_v2.py `
  --checkpoint results\v2_large_dataset\checkpoints\structured_pipeline.pt `
  --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna `
  --output-dir results\v2_large_evaluation `
  --max-records 20 `
  --device auto
```

This generates:
- `evaluation_report.json` - metrics for model vs baseline
- `evaluation_report.md` - human-readable summary
- `midi/*.mid` - generated MIDI files

**Expected improvements over small model:**

| Metric | Small Model | Large Model (Target) |
|--------|-------------|----------------------|
| Pitch range | 14.6 | 25+ |
| Unique pitches | 10.8 | 18+ |
| Pitch entropy | 2.86 | 3.2+ |
| Chord-tone ratio | 60% | 70%+ |
| Self-similarity | 0.14 | <0.10 |

---

## 🔬 Step 6: Ablation Study (Optional)

Verify that bio conditioning actually works:

### 6.1 Train "No Bio" Baseline

```json
// Create configs/pipeline_v2_large_no_bio.json
// Set "embedding_dim": 0 in "bio" section
```

```powershell
.\.venv\Scripts\python.exe train_bio_music_v2.py `
  --config configs\pipeline_v2_large_no_bio.json
```

### 6.2 Train "Random Bio" Baseline

```python
# Modify bio_music_pipeline/v2/structured_train.py
# Replace bio_vector with random noise:
# bio_vector = torch.randn_like(bio_vector)
```

### 6.3 Compare Results

```powershell
# Generate from same genome with all 3 models
# Compare metrics - real_bio should be best
```

If `real_bio` is NOT better than `random_bio`, bio conditioning is not working.

---

## 🐛 Troubleshooting

### Out of Memory

```json
// Reduce in configs/pipeline_v2_large.json:
"batch_size": 16,  // was 32
"d_model": 384,    // was 512
"n_layers": 6,     // was 8
```

### Training Too Slow

```json
// Increase:
"num_workers": 8,  // was 4
"batch_size": 64,  // was 32
```

### Loss Not Decreasing

```json
// Try different learning rate:
"learning_rate": 0.0001,  // was 0.0003
// Or add warmup (requires code change)
```

### ESM Embedding Fails

```json
// Disable ESM if causing issues:
"use_esm_embedding": false,
"embedding_dim": 256  // was 384
```

---

## 📁 Output Structure

After training, you'll have:

```
results/v2_large_dataset/
├── checkpoints/
│   ├── structured_pipeline.pt  (full pipeline)
│   ├── harmony_best.pt         (best harmony model)
│   └── melody_best.pt          (best melody model)
├── metrics.json                (training history)
├── resolved_config.json        (actual config used)
├── pairing/
│   └── pair_manifest.json      (bio-music pairs)
└── smoke/
    └── structured_sample.mid   (test generation)
```

---

## 🎯 Next Steps

1. **Compare with small model:**
   - Generate from same FASTA with both checkpoints
   - Compare metrics side-by-side

2. **Try different organisms:**
   - Generate music from all 5 genomes
   - Check if different organisms produce different music

3. **Visualize bio-music correlation:**
   - Plot bio features vs music features
   - t-SNE of bio embeddings colored by organism

4. **Write paper:**
   - Document architecture
   - Report metrics
   - Include generated examples

---

## 📚 Related Files

- [DATA_SUMMARY.md](../DATA_SUMMARY.md) - Dataset statistics
- [README.md](../README.md) - Project overview
- [docs/architecture_and_science.md](../docs/architecture_and_science.md) - Technical details
- [RUN_FROM_SCRATCH.md](../RUN_FROM_SCRATCH.md) - Full setup guide

---

## ⚡ Quick Commands Reference

```powershell
# Dataset report
.\.venv\Scripts\python.exe tools\report_structured_dataset.py --config configs\pipeline_v2_large.json --output-dir results\v2_large_dataset_report

# Train
.\.venv\Scripts\python.exe train_bio_music_v2.py --config configs\pipeline_v2_large.json

# Generate
.\.venv\Scripts\python.exe generate_from_fasta_v2.py --config configs\pipeline_v2_large.json --checkpoint results\v2_large_dataset\checkpoints\structured_pipeline.pt --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna --output results\v2_large_generation\output.mid

# Evaluate
.\.venv\Scripts\python.exe tools\evaluate_structured_v2.py --checkpoint results\v2_large_dataset\checkpoints\structured_pipeline.pt --fasta data\fasta\refseq_genomes\GCF_000005845.2_genomic.fna --output-dir results\v2_large_evaluation --max-records 20
```
