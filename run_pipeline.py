#!/usr/bin/env python3
"""
Bio-Music Pipeline - Main Orchestration Script

Reproducible controlled symbolic music generation conditioned on biological sequence features.

This script executes all 5 stages of the pipeline:
1. Bio-vector extraction from genomic sequences
2. Deterministic sonification rule application
3. Music dataset preparation with strict splits
4. Bio-conditioned transformer training
5. Generation and comprehensive evaluation

IMPORTANT: This system demonstrates bio-vectors as structured conditioning signals.
No causal relationship between genes and music is claimed or implied.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm

# Add package to path
sys.path.insert(0, str(Path(__file__).parent))

from bio_music_pipeline import set_seed
from bio_music_pipeline.extractors import BioVectorExtractor
from bio_music_pipeline.sonification import SonificationMapper, apply_sonification_rules
from bio_music_pipeline.data import MusicDataset, MIDIPreprocessor
from bio_music_pipeline.models import create_bio_music_model, BioConditionedTransformerDecoder
from bio_music_pipeline.baselines import create_baselines, MarkovBaseline
from bio_music_pipeline.evaluation import (
    StatisticalValidator, 
    InformationTransferClassifier,
    HumanEvaluationSurvey,
    run_comprehensive_evaluation
)
from bio_music_pipeline.utils import (
    check_gradient_flow,
    verify_no_data_leak,
    create_sample_bio_sequences,
    batch_tokens_to_midi
)


class BioMusicPipeline:
    """Main pipeline orchestrator."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set seeds for reproducibility
        set_seed(self.config['pipeline']['seed'])
        
        # Setup directories
        self.output_dir = Path(self.config['pipeline']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Results tracking
        self.results = {
            'config': self.config,
            'start_time': datetime.now().isoformat(),
            'stages_completed': [],
            'metrics': {},
            'hypothesis_tests': {},
            'artifacts': {}
        }
        
        print("=" * 60)
        print("BIO-MUSIC CONDITIONED GENERATION PIPELINE")
        print("=" * 60)
        print("\nDISCLAIMER: This system uses bio-vectors as structured")
        print("conditioning signals. NO causal relationship between")
        print("genes and music is claimed or demonstrated.\n")
    
    def stage1_extract_bio_vectors(self, sequences=None):
        """Stage 1: Extract bio-vectors from biological sequences."""
        print("\n" + "=" * 60)
        print("STAGE 1: Bio-Vector Extraction")
        print("=" * 60)
        
        extractor_config = self.config['extraction']
        extractor = BioVectorExtractor(
            kmer_sizes=extractor_config['kmer_sizes'],
            window_size=extractor_config['window_size'],
            stride=extractor_config['stride'],
            min_sequence_length=extractor_config['min_sequence_length']
        )
        
        if sequences is None:
            # Generate synthetic sequences for demonstration
            print("Generating synthetic biological sequences...")
            sequences = create_sample_bio_sequences(
                n_samples=50,
                length_range=(500, 2000),
                seed=self.config['pipeline']['seed']
            )
        
        target_dim = self.config['model']['bio_vector_dim']
        bio_vectors = []
        sequence_ids = []
        
        for i, seq in enumerate(tqdm(sequences, desc="Extracting bio-vectors")):
            seq_clean = ''.join([c for c in seq.upper() if c in 'ACGT'])
            if len(seq_clean) >= extractor.min_sequence_length:
                features = extractor.extract_features(seq_clean)
                vector = extractor.create_bio_vector(features, target_dim)
                bio_vectors.append(vector)
                sequence_ids.append(f"seq_{i:04d}")
        
        bio_vectors = np.array(bio_vectors)
        
        # Save bio-vectors
        bio_vectors_path = self.output_dir / 'bio_vectors.npy'
        np.save(bio_vectors_path, bio_vectors)
        
        # Save sequence IDs
        with open(self.output_dir / 'sequence_ids.json', 'w') as f:
            json.dump(sequence_ids, f)
        
        self.results['stage1'] = {
            'n_sequences': len(bio_vectors),
            'bio_vector_dim': bio_vectors.shape[1],
            'feature_stats': {
                'mean': float(bio_vectors.mean()),
                'std': float(bio_vectors.std()),
                'min': float(bio_vectors.min()),
                'max': float(bio_vectors.max())
            }
        }
        self.results['artifacts']['bio_vectors'] = str(bio_vectors_path)
        self.results['stages_completed'].append('stage1_extraction')
        
        print(f"\nExtracted {len(bio_vectors)} bio-vectors of dimension {target_dim}")
        print(f"Saved to: {bio_vectors_path}")
        
        return bio_vectors, sequence_ids
    
    def stage2_apply_sonification(self, bio_vectors: np.ndarray):
        """Stage 2: Apply deterministic sonification rules."""
        print("\n" + "=" * 60)
        print("STAGE 2: Sonification Rule Application")
        print("=" * 60)
        
        sonification_config = self.config['sonification']
        mapper = SonificationMapper(
            tempo_range=tuple(sonification_config['tempo_range']),
            pitch_range=tuple(sonification_config['pitch_range']),
            key_mapping=sonification_config['key_mapping'],
            chord_complexity_levels=sonification_config['chord_complexity_levels']
        )
        
        # Convert bio-vectors to musical parameters
        musical_params_list = []
        conditioning_vectors = []
        
        for bio_vec in tqdm(bio_vectors, desc="Applying sonification rules"):
            params = mapper.bio_vector_to_musical_params(bio_vec)
            musical_params_list.append(params)
            cond_vec = mapper.create_conditioning_vector(params)
            conditioning_vectors.append(cond_vec)
        
        conditioning_vectors = np.array(conditioning_vectors)
        
        # Save conditioning vectors
        cond_path = self.output_dir / 'conditioning_vectors.npy'
        np.save(cond_path, conditioning_vectors)
        
        # Save musical parameters summary
        params_summary = []
        for i, params in enumerate(musical_params_list[:10]):  # Sample
            params_summary.append({
                'index': i,
                'key': params.key,
                'tempo': params.tempo,
                'pitch_range': params.pitch_range,
                'rhythm_complexity': params.rhythm_complexity,
                'scale_type': params.scale_type
            })
        
        with open(self.output_dir / 'musical_params_sample.json', 'w') as f:
            json.dump(params_summary, f, indent=2)
        
        self.results['stage2'] = {
            'n_samples': len(conditioning_vectors),
            'conditioning_dim': conditioning_vectors.shape[1],
            'key_distribution': self._compute_key_distribution(musical_params_list),
            'tempo_range_actual': [
                float(min(p.tempo for p in musical_params_list)),
                float(max(p.tempo for p in musical_params_list))
            ]
        }
        self.results['artifacts']['conditioning_vectors'] = str(cond_path)
        self.results['stages_completed'].append('stage2_sonification')
        
        print(f"\nGenerated {len(conditioning_vectors)} conditioning vectors")
        print(f"Dimension: {conditioning_vectors.shape[1]}")
        print(f"Saved to: {cond_path}")
        
        return conditioning_vectors, musical_params_list
    
    def _compute_key_distribution(self, params_list):
        """Compute distribution of musical keys."""
        from collections import Counter
        keys = [p.key for p in params_list]
        counts = Counter(keys)
        total = len(keys)
        return {k: round(v/total, 3) for k, v in counts.items()}
    
    def stage3_prepare_dataset(self, midi_dir: str = None):
        """Stage 3: Prepare music dataset with strict splits."""
        print("\n" + "=" * 60)
        print("STAGE 3: Music Dataset Preparation")
        print("=" * 60)
        
        data_config = self.config['data']
        model_config = self.config['model']
        
        # Create sample MIDI data if directory not provided
        if midi_dir is None or not Path(midi_dir).exists():
            print("Creating synthetic MIDI training data...")
            midi_dir = self.output_dir / 'synthetic_midi'
            midi_dir.mkdir(parents=True, exist_ok=True)
            self._create_synthetic_midi_data(midi_dir, n_files=100)
        
        # Initialize dataset
        dataset = MusicDataset(
            data_dir=midi_dir,
            train_split=data_config['train_split'],
            val_split=data_config['val_split'],
            test_split=data_config['test_split'],
            seed=self.config['pipeline']['seed'],
            max_seq_len=model_config['max_seq_len'],
            min_duration=data_config['min_duration'],
            max_duration=data_config['max_duration']
        )
        
        # Save split information
        splits_dir = self.output_dir / 'data_splits'
        dataset.save_splits(str(splits_dir))
        
        # Verify no data leak
        train_files = set(d[0] for d in dataset.train_data)
        test_files = set(d[0] for d in dataset.test_data)
        no_leak = verify_no_data_leak(train_files, test_files)
        
        self.results['stage3'] = {
            'n_train': len(dataset.train_data),
            'n_val': len(dataset.val_data),
            'n_test': len(dataset.test_data),
            'vocab_size': dataset.preprocessor.vocab_size,
            'no_data_leak': no_leak,
            'splits_directory': str(splits_dir)
        }
        self.results['stages_completed'].append('stage3_dataset')
        
        print(f"\nDataset prepared:")
        print(f"  Train: {len(dataset.train_data)} samples")
        print(f"  Val: {len(dataset.val_data)} samples")
        print(f"  Test: {len(dataset.test_data)} samples")
        print(f"  Vocabulary size: {dataset.preprocessor.vocab_size}")
        print(f"  Data leak check: {'PASSED' if no_leak else 'FAILED'}")
        
        return dataset
    
    def _create_synthetic_midi_data(self, output_dir: Path, n_files: int = 100):
        """Create synthetic MIDI files for training."""
        import random
        
        for i in range(n_files):
            midi = mido.MidiFile()
            midi.ticks_per_beat = 480
            track = mido.MidiTrack()
            midi.tracks.append(track)
            
            # Random musical parameters
            tempo = random.randint(80, 160)
            track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))
            
            # Generate random melody
            current_time = 0
            key = random.choice(['C', 'D', 'E', 'F', 'G', 'A', 'B'])
            scale = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
            
            for _ in range(random.randint(50, 200)):
                note = 60 + random.choice(scale) + random.randint(-12, 12)
                note = max(48, min(96, note))
                velocity = random.randint(60, 100)
                duration = random.randint(100, 500)
                
                track.append(mido.Message('note_on', note=note, velocity=velocity, time=random.randint(0, 200)))
                track.append(mido.Message('note_off', note=note, velocity=0, time=duration))
            
            track.append(mido.MetaMessage('end_of_track', time=0))
            midi.save(output_dir / f'synthetic_{i:04d}.mid')
    
    def stage4_train_model(self, dataset: MusicDataset, bio_vectors: np.ndarray):
        """Stage 4: Train bio-conditioned transformer model."""
        print("\n" + "=" * 60)
        print("STAGE 4: Model Training")
        print("=" * 60)
        
        model_config = self.config['model']
        training_config = self.config['training']
        
        # Create model
        model = create_bio_music_model(model_config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        print(f"Model created on {device}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Verify gradient flow
        print("\nVerifying gradient flow...")
        dummy_tokens = torch.randint(0, model_config['vocab_size'], 
                                     (10, 4)).to(device)  # (seq_len, batch)
        dummy_bio = torch.randn(4, model_config['bio_vector_dim']).to(device)
        
        grad_check = check_gradient_flow(model, {
            'tokens': dummy_tokens,
            'bio_vectors': dummy_bio
        })
        
        if not grad_check['flow_verified']:
            print("WARNING: Gradient flow issues detected!")
        else:
            print("Gradient flow verified ✓")
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        
        # Training loop
        print("\nStarting training...")
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(training_config['epochs']):
            model.train()
            total_loss = 0
            n_batches = 0
            
            for batch in dataset.get_train_loader(batch_size=training_config['batch_size']):
                # Prepare batch
                token_sequences = [item[1] for item in batch]
                max_len = max(len(seq) for seq in token_sequences)
                
                # Pad sequences
                tokens_padded = []
                for seq in token_sequences:
                    padded = seq + [1] * (max_len - len(seq))  # Pad with EOS
                    tokens_padded.append(padded)
                
                tokens = torch.tensor(tokens_padded, dtype=torch.long).to(device)
                tokens = tokens.transpose(0, 1)  # (seq_len, batch)
                
                # Sample bio-vectors (cycle through if needed)
                bio_idx = [n_batches % len(bio_vectors) for n_batches in range(tokens.size(1))]
                bio_batch = torch.tensor(bio_vectors[bio_idx], dtype=torch.float32).to(device)
                
                # Forward pass
                loss_dict = model.compute_loss(
                    tokens, 
                    bio_batch,
                    aux_loss_weight=training_config['aux_loss_weight']
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_config['grad_clip'])
                
                optimizer.step()
                model.update_temperature()  # Anneal Gumbel temperature
                
                total_loss += loss_dict['total_loss'].item()
                n_batches += 1
            
            avg_train_loss = total_loss / max(n_batches, 1)
            
            # Validation
            model.eval()
            val_loss = 0
            n_val_batches = 0
            
            with torch.no_grad():
                for batch in dataset.get_val_loader(batch_size=training_config['batch_size']):
                    token_sequences = [item[1] for item in batch]
                    if len(token_sequences) == 0:
                        continue
                    
                    max_len = max(len(seq) for seq in token_sequences)
                    tokens_padded = []
                    for seq in token_sequences:
                        padded = seq + [1] * (max_len - len(seq))
                        tokens_padded.append(padded)
                    
                    tokens = torch.tensor(tokens_padded, dtype=torch.long).to(device)
                    tokens = tokens.transpose(0, 1)
                    
                    bio_idx = [n_val_batches % len(bio_vectors) for n_val_batches in range(tokens.size(1))]
                    bio_batch = torch.tensor(bio_vectors[bio_idx], dtype=torch.float32).to(device)
                    
                    loss_dict = model.compute_loss(tokens, bio_batch, aux_loss_weight=0.0)
                    val_loss += loss_dict['total_loss'].item()
                    n_val_batches += 1
            
            avg_val_loss = val_loss / max(n_val_batches, 1)
            
            print(f"Epoch {epoch+1}/{training_config['epochs']}: "
                  f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                
                # Save best model
                model_path = self.output_dir / 'models' / 'best_model.pt'
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'config': model_config
                }, model_path)
            else:
                patience_counter += 1
                if patience_counter >= training_config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.results['stage4'] = {
            'final_train_loss': float(avg_train_loss),
            'best_val_loss': float(best_val_loss),
            'epochs_trained': epoch + 1,
            'gradient_flow_verified': grad_check['flow_verified'],
            'model_parameters': sum(p.numel() for p in model.parameters())
        }
        self.results['artifacts']['best_model'] = str(model_path)
        self.results['stages_completed'].append('stage4_training')
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: {model_path}")
        
        return model
    
    def stage5_generate_and_evaluate(self, 
                                     model: BioConditionedTransformerDecoder,
                                     dataset: MusicDataset,
                                     bio_vectors: np.ndarray,
                                     conditioning_vectors: np.ndarray):
        """Stage 5: Generate samples and run comprehensive evaluation."""
        print("\n" + "=" * 60)
        print("STAGE 5: Generation & Evaluation")
        print("=" * 60)
        
        model_config = self.config['model']
        eval_config = self.config['evaluation']
        device = next(model.parameters()).device
        
        # Load best model weights
        model_path = self.output_dir / 'models' / 'best_model.pt'
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Create baselines
        print("\nCreating baseline generators...")
        baselines = create_baselines(model_config)
        
        # Fit Markov baseline on training data
        print("Fitting Markov baseline...")
        train_sequences = [[item[1] for item in batch] 
                          for batch in dataset.get_train_loader(batch_size=32)]
        flat_train = [seq for batch in train_sequences for seq in batch]
        baselines['markov'].fit(flat_train)
        
        # Generate samples
        print("\nGenerating samples...")
        n_samples = eval_config['n_samples_per_bio']
        max_len = model_config['max_seq_len']
        
        generated_samples = {}
        midi_output_dir = self.output_dir / 'midi'
        midi_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Conditioned generation
        print("  - Bio-conditioned generation...")
        bio_subset = bio_vectors[:n_samples]
        bio_tensor = torch.tensor(bio_subset, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            conditioned_seqs = model.generate(
                bio_tensor, 
                max_len=max_len,
                temperature=1.0,
                use_gumbel=False
            )
        generated_samples['conditioned'] = conditioned_seqs.cpu().numpy()
        
        # Unconditional generation
        print("  - Unconditional generation...")
        dummy_bio = torch.zeros_like(bio_tensor)
        with torch.no_grad():
            unconditional_seqs = model.generate(
                dummy_bio,
                max_len=max_len,
                temperature=1.0,
                use_gumbel=False
            )
        generated_samples['unconditional'] = unconditional_seqs.cpu().numpy()
        
        # Baseline generations
        print("  - Random baseline...")
        generated_samples['random'] = baselines['random'].generate(
            n_samples, max_len).numpy()
        
        print("  - Markov baseline...")
        generated_samples['markov'] = baselines['markov'].generate(
            n_samples, max_len).numpy()
        
        print("  - Rule-based baseline...")
        from bio_music_pipeline.sonification import SonificationMapper
        mapper = SonificationMapper()
        musical_params = mapper.bio_vector_to_musical_params(bio_vectors[0])
        generated_samples['rule_based'] = baselines['rule_based'].generate_from_params(
            vars(musical_params), n_samples, max_len).numpy()
        
        # Save MIDI files
        print("\nSaving MIDI files...")
        vocab = dataset.preprocessor.token_to_idx
        
        for condition, sequences in generated_samples.items():
            condition_dir = midi_output_dir / condition
            condition_dir.mkdir(parents=True, exist_ok=True)
            
            filenames = [f"{condition}_{i:04d}.mid" for i in range(len(sequences))]
            created_files = batch_tokens_to_midi(
                torch.tensor(sequences),
                vocab,
                str(condition_dir),
                filenames=filenames
            )
            print(f"  {condition}: {len(created_files)} MIDI files saved")
        
        # Run evaluation
        print("\nRunning comprehensive evaluation...")
        eval_results = run_comprehensive_evaluation(
            eval_config,
            dataset.get_test_data(),
            generated_samples,
            bio_vectors,
            str(self.output_dir / 'reports')
        )
        
        # Generate human evaluation survey
        print("\nGenerating human evaluation survey...")
        survey_generator = HumanEvaluationSurvey(str(self.output_dir / 'surveys'))
        
        midi_files_dict = {}
        for condition in generated_samples.keys():
            condition_dir = midi_output_dir / condition
            midi_files_dict[condition] = [
                str(f) for f in condition_dir.glob("*.mid")
            ][:5]  # Limit to 5 per condition
        
        survey_path = survey_generator.generate_survey_html(midi_files_dict)
        
        # Compile final results
        self.results['stage5'] = {
            'n_samples_generated': {k: len(v) for k, v in generated_samples.items()},
            'midi_files_directory': str(midi_output_dir),
            'survey_path': survey_path,
            'baseline_comparison': self._compare_baselines(generated_samples)
        }
        self.results['hypothesis_tests'] = eval_results
        self.results['artifacts']['midi_files'] = str(midi_output_dir)
        self.results['artifacts']['survey'] = survey_path
        self.results['stages_completed'].append('stage5_evaluation')
        
        print(f"\nEvaluation complete!")
        print(f"MIDI files saved to: {midi_output_dir}")
        print(f"Survey generated: {survey_path}")
        
        return generated_samples, eval_results
    
    def _compare_baselines(self, generated_samples: dict) -> dict:
        """Compare quality metrics across baselines."""
        comparison = {}
        for name, samples in generated_samples.items():
            comparison[name] = {
                'mean_token_value': float(np.mean(samples)),
                'std_token_value': float(np.std(samples)),
                'unique_tokens': float(np.mean([len(set(s)) for s in samples]))
            }
        return comparison
    
    def save_final_report(self):
        """Save comprehensive final report."""
        print("\n" + "=" * 60)
        print("Saving Final Report")
        print("=" * 60)
        
        self.results['end_time'] = datetime.now().isoformat()
        self.results['status'] = 'completed'
        
        # Compute hypothesis statuses
        hypothesis_status = {}
        
        if 'disentanglement' in self.results.get('hypothesis_tests', {}):
            disentangle = self.results['hypothesis_tests']['disentanglement']
            hypothesis_status['H1_disentanglement'] = {
                'description': 'Conditioned generation outperforms unconditional',
                'significant': disentangle.get('significant', False),
                'effect_size': disentangle.get('cohens_d', 0),
                'p_value': disentangle.get('p_value', 1.0)
            }
        
        if 'permutation_test' in self.results.get('hypothesis_tests', {}):
            perm_test = self.results['hypothesis_tests']['permutation_test']
            hypothesis_status['H2_information_transfer'] = {
                'description': 'Bio-cluster information recoverable from music',
                'significant': perm_test.get('significant', False),
                'accuracy': perm_test.get('original_accuracy', 0),
                'p_value': perm_test.get('permutation_p_value', 1.0)
            }
        
        self.results['hypothesis_status'] = hypothesis_status
        
        # Save JSON report
        report_path = self.output_dir / 'final_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save summary
        summary_path = self.output_dir / 'summary.txt'
        with open(summary_path, 'w') as f:
            f.write("BIO-MUSIC PIPELINE EXECUTION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Status: {self.results['status']}\n")
            f.write(f"Start: {self.results['start_time']}\n")
            f.write(f"End: {self.results['end_time']}\n\n")
            f.write("Stages Completed:\n")
            for stage in self.results['stages_completed']:
                f.write(f"  ✓ {stage}\n")
            f.write("\nHypothesis Tests:\n")
            for h_name, h_result in hypothesis_status.items():
                status = "SUPPORTED" if h_result['significant'] else "NOT SUPPORTED"
                f.write(f"  {h_name}: {status} (p={h_result['p_value']:.4f})\n")
            f.write("\nArtifacts:\n")
            for name, path in self.results['artifacts'].items():
                f.write(f"  {name}: {path}\n")
        
        print(f"\nFinal report saved to: {report_path}")
        print(f"Summary saved to: {summary_path}")
        
        return self.results
    
    def run(self, sequences=None, midi_dir=None):
        """Execute complete pipeline."""
        try:
            # Stage 1
            bio_vectors, sequence_ids = self.stage1_extract_bio_vectors(sequences)
            
            # Stage 2
            conditioning_vectors, musical_params = self.stage2_apply_sonification(bio_vectors)
            
            # Stage 3
            dataset = self.stage3_prepare_dataset(midi_dir)
            
            # Stage 4
            model = self.stage4_train_model(dataset, bio_vectors)
            
            # Stage 5
            generated_samples, eval_results = self.stage5_generate_and_evaluate(
                model, dataset, bio_vectors, conditioning_vectors
            )
            
            # Save report
            final_results = self.save_final_report()
            
            print("\n" + "=" * 60)
            print("PIPELINE EXECUTION COMPLETE")
            print("=" * 60)
            print(f"\nAll results saved to: {self.output_dir}")
            print("\nKey artifacts:")
            for name, path in final_results['artifacts'].items():
                print(f"  - {name}: {path}")
            
            return final_results
            
        except Exception as e:
            print(f"\nERROR: Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            self.results['status'] = 'failed'
            self.results['error'] = str(e)
            return self.results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Bio-Music Conditioned Generation Pipeline'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='configs/pipeline_config.json',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--sequences',
        type=str,
        default=None,
        help='Path to FASTA file with biological sequences (optional)'
    )
    parser.add_argument(
        '--midi-dir',
        type=str,
        default=None,
        help='Path to directory with MIDI training data (optional)'
    )
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = BioMusicPipeline(args.config)
    results = pipeline.run(midi_dir=args.midi_dir)
    
    # Exit with appropriate code
    if results.get('status') == 'completed':
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
