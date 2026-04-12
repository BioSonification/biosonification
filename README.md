# Bio-Music Conditioned Generation Pipeline

A reproducible pipeline for controlled symbolic music generation conditioned on biological sequence features.

## Important Disclaimer

**This system demonstrates bio-vectors as structured conditioning signals for music generation. NO causal relationship between genes and music is claimed, implied, or demonstrated.** The bio-vectors serve as deterministic control signals that statistically influence musical structure, ensuring consistency when reusing the same input and outperforming unconditional baselines.

## Overview

The pipeline consists of five sequential stages:

1. **Bio-Vector Extraction**: Streaming extraction of fixed-dimensional feature vectors from genomic sequences (nucleotide frequencies, entropy, k-mer distributions, shift statistics)

2. **Sonification Rules**: Deterministic, transparent mapping of bio-vectors to explicit musical parameters (key, tempo, pitch range, rhythm complexity, chord distribution)

3. **Dataset Preparation**: Instrumental music dataset with strict non-overlapping train/val/test splits

4. **Model Training**: Autoregressive transformer decoder with bio-conditioning, Gumbel-Softmax differentiable sampling, and auxiliary language model loss

5. **Generation & Evaluation**: Comprehensive evaluation including disentanglement tests, correlation analysis, permutation tests, and human evaluation survey

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Run complete pipeline with default configuration
python run_pipeline.py --config configs/pipeline_config.json
```

### With Your Own Data

The pipeline supports loading your own MIDI and FASTA datasets from any directory.

#### 1. Setup Data Directories

```bash
# Create standard data directories
python scan_datasets.py --setup --project-root /workspace
```

This creates:
- `data/midi/` - Place your MIDI files here
- `data/fasta/` - Place your FASTA files here

#### 2. Add Your Data

**MIDI Files:**
- Download from any source (MuseScore, MIDI World, BitMidi, etc.)
- Place in `data/midi/` or any subdirectory
- Supported formats: `.mid`, `.midi`

**FASTA Files:**
- Download from NCBI, Ensembl, UCSC Genome Browser, UniProt, etc.
- Place in `data/fasta/` or any subdirectory
- Supported formats: `.fasta`, `.fa`, `.fna`, `.ffn`, `.faa`, `.frn`

#### 3. Scan for Data

```bash
# Verify your files are detected
python scan_datasets.py
```

#### 4. Run Pipeline

```bash
# Using your data in the standard directories
python run_pipeline.py --config configs/pipeline_config.json

# Or specify custom directories via config
python run_pipeline.py --config configs/data_paths_config.json
```

### Command-Line Examples

```bash
# Using custom biological sequences (FASTA format)
python run_pipeline.py --sequences data/sequences.fasta --midi-dir data/midi_files

# Using custom MIDI training data only
python run_pipeline.py --midi-dir /path/to/midi/training/data

# Scan specific directory for data
python scan_datasets.py --path /path/to/your/data
```

## Configuration

Edit `configs/pipeline_config.json` to customize:

- Model architecture (dimensions, layers, heads)
- Training parameters (learning rate, batch size, epochs)
- Extraction settings (k-mer sizes, window size)
- Sonification rules (tempo range, pitch range)
- Evaluation settings (significance level, permutations)

## Output Structure

```
results/
├── bio_vectors.npy          # Extracted bio-feature vectors
├── conditioning_vectors.npy  # Musical parameter vectors
├── best_model.pt            # Trained model weights
├── midi/                    # Generated MIDI files
│   ├── conditioned/
│   ├── unconditional/
│   ├── random/
│   ├── markov/
│   └── rule_based/
├── reports/
│   └── evaluation_results.json
├── surveys/
│   └── human_evaluation_survey.html
├── final_report.json        # Complete results
└── summary.txt              # Human-readable summary
```

## Key Features

### Reproducibility
- Fixed random seeds across all libraries
- Deterministic behavior in cuDNN
- Version-locked dependencies

### Gradient Flow Verification
- Gumbel-Softmax with straight-through estimation
- Auxiliary LM with frozen weights but flowing gradients
- Built-in gradient checking utilities

### Data Integrity
- Strict non-overlapping train/val/test splits
- Automatic data leak detection
- Test data excluded from all training decisions

### Statistical Validation
- Disentanglement gap measurement
- Permutation tests for significance
- Noise robustness checks
- Classifier-based information transfer verification

## Architecture Details

### Bio-Conditioned Transformer Decoder

```
Bio-vector (128 dim) → BioConditioningModule → Transformer Decoder → Music Tokens
                              ↓
                    Musical Parameters
                    (key, tempo, range, etc.)
```

### Composite Loss Function

```
Total Loss = CrossEntropy(predictions, targets) 
           + λ × AuxiliaryLM(hidden_states)
```

Where the auxiliary LM has frozen weights but allows gradient flow.

## Evaluation Metrics

1. **Disentanglement Gap**: Statistical difference between conditioned and unconditional generation quality

2. **Information Transfer**: Accuracy of recovering bio-vector clusters from generated music features

3. **Correlation Analysis**: Significant correlations between bio-features and musical parameters with noise robustness checks

4. **Human Evaluation**: Blind survey with attention checks and reliability metrics

## Baselines

- **Random**: Uniform random token sampling
- **Markov**: N-gram Markov chain trained on data
- **Unconditional**: Same architecture without bio-conditioning
- **Rule-Based**: Direct application of sonification rules

## API Usage

```python
from bio_music_pipeline import set_seed
from bio_music_pipeline.extractors import BioVectorExtractor
from bio_music_pipeline.sonification import SonificationMapper
from bio_music_pipeline.models import create_bio_music_model

# Set seed for reproducibility
set_seed(42)

# Extract bio-vectors
extractor = BioVectorExtractor(kmer_sizes=[1, 2, 3])
sequences = ["ACGTACGT...", "GCTAGCTA..."]
bio_vectors = extractor.extract_features(sequences[0])

# Apply sonification
mapper = SonificationMapper()
musical_params = mapper.bio_vector_to_musical_params(bio_vectors)

# Create model
model = create_bio_music_model(config)
```

## Citation

If you use this pipeline in your research, please note:

> This system demonstrates bio-vectors as structured conditioning signals for music generation. No biological causality is claimed between genetic sequences and musical output.

## License

MIT License

## Contributing

Contributions welcome! Please ensure:
- All tests pass
- Code follows existing style
- Documentation is updated
- Random seeds are set for reproducibility
