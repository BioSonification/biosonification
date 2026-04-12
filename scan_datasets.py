#!/usr/bin/env python3
"""
Script to scan and display available MIDI and FASTA datasets.

This script helps you verify that your data files are properly placed
and will be detected by the pipeline.
"""

import argparse
from pathlib import Path
from bio_music_pipeline.data import UniversalDataLoader, setup_user_datasets


def main():
    parser = argparse.ArgumentParser(
        description='Scan for MIDI and FASTA datasets in the project'
    )
    parser.add_argument(
        '--path', 
        type=str, 
        default=None,
        help='Path to search (defaults to project root)'
    )
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Setup standard data directories'
    )
    parser.add_argument(
        '--project-root',
        type=str,
        default=None,
        help='Project root directory for --setup'
    )
    
    args = parser.parse_args()
    
    # Setup directories if requested
    if args.setup:
        project_root = args.project_root or '.'
        setup_user_datasets(project_root)
        print()
    
    # Scan and report
    loader = UniversalDataLoader()
    search_path = args.path or '.'
    
    print(f"Scanning for data files in: {Path(search_path).resolve()}")
    report = loader.scan_and_report(search_path)
    loader.print_data_summary(search_path)
    
    # Provide next steps
    print("\nNEXT STEPS:")
    print("-" * 60)
    
    if report['total_midi_count'] == 0:
        print("No MIDI files found.")
        print("→ Place your .mid or .midi files in: data/midi/")
        print("→ Or create a subdirectory: data/midi/your_collection/")
    else:
        print(f"✓ Found {report['total_midi_count']} MIDI file(s)")
        print("→ These will be automatically used by the pipeline")
    
    if report['total_fasta_count'] == 0:
        print("No FASTA files found.")
        print("→ Place your .fasta or .fa files in: data/fasta/")
        print("→ Or create a subdirectory: data/fasta/your_genomes/")
    else:
        print(f"✓ Found {report['total_fasta_count']} FASTA file(s)")
        print("→ These will be automatically used by the pipeline")
    
    print("\nTo run the full pipeline:")
    print("  python run_pipeline.py --config configs/pipeline_config.json")
    print("\nTo use custom data paths, edit:")
    print("  configs/data_paths_config.json")
    print("-" * 60)


if __name__ == '__main__':
    main()
