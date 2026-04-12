"""
Universal Data Loader for MIDI and FASTA files.

This module provides flexible loading of user-provided datasets
from any directory within the project structure.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class DataLoaderConfig:
    """Configuration for data loading."""
    midi_dir: Optional[str] = None
    fasta_dir: Optional[str] = None
    min_midi_duration: float = 30.0
    max_midi_duration: float = 300.0
    min_sequence_length: int = 100
    supported_midi_extensions: Tuple[str, ...] = ('.mid', '.midi')
    supported_fasta_extensions: Tuple[str, ...] = ('.fasta', '.fa', '.fna', '.ffn', '.faa', '.frn')


class UniversalDataLoader:
    """
    Universal loader for MIDI and FASTA datasets.
    
    Supports loading from any directory within the project structure.
    Users can place their data files in any subdirectory and specify
    the path via configuration or command-line arguments.
    """
    
    def __init__(self, config: Optional[DataLoaderConfig] = None):
        """
        Initialize the universal data loader.
        
        Args:
            config: DataLoaderConfig instance or None for defaults
        """
        self.config = config or DataLoaderConfig()
        self.project_root = Path(__file__).parent.parent.parent
        
    def find_data_directories(self, search_path: Optional[str] = None) -> Dict[str, List[Path]]:
        """
        Find all directories containing MIDI or FASTA files.
        
        Args:
            search_path: Starting path for search (defaults to project root)
            
        Returns:
            Dictionary with 'midi_dirs' and 'fasta_dirs' lists
        """
        if search_path is None:
            search_path = str(self.project_root)
        
        search_path = Path(search_path)
        midi_dirs = set()
        fasta_dirs = set()
        
        # Search for MIDI files
        for ext in self.config.supported_midi_extensions:
            for filepath in search_path.rglob(f"*{ext}"):
                midi_dirs.add(filepath.parent)
        
        # Search for FASTA files
        for ext in self.config.supported_fasta_extensions:
            for filepath in search_path.rglob(f"*{ext}"):
                fasta_dirs.add(filepath.parent)
        
        return {
            'midi_dirs': sorted(list(midi_dirs)),
            'fasta_dirs': sorted(list(fasta_dirs))
        }
    
    def validate_directory(self, directory: str, file_type: str = 'auto') -> bool:
        """
        Validate that a directory contains valid data files.
        
        Args:
            directory: Path to directory
            file_type: 'midi', 'fasta', or 'auto' for automatic detection
            
        Returns:
            True if directory contains valid files
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            print(f"Warning: Directory does not exist: {directory}")
            return False
        
        if not dir_path.is_dir():
            print(f"Warning: Path is not a directory: {directory}")
            return False
        
        if file_type == 'auto' or file_type == 'midi':
            for ext in self.config.supported_midi_extensions:
                if list(dir_path.rglob(f"*{ext}")):
                    if file_type == 'midi':
                        return True
        
        if file_type == 'auto' or file_type == 'fasta':
            for ext in self.config.supported_fasta_extensions:
                if list(dir_path.rglob(f"*{ext}")):
                    if file_type == 'fasta':
                        return True
        
        if file_type == 'auto':
            # For auto, return True if any valid files found
            return bool(list(dir_path.rglob("*.mid")) or 
                       list(dir_path.rglob("*.midi")) or
                       list(dir_path.rglob("*.fasta")) or
                       list(dir_path.rglob("*.fa")))
        
        return False
    
    def get_midi_files(self, directory: str, recursive: bool = True) -> List[Path]:
        """
        Get all MIDI files from a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to search recursively
            
        Returns:
            List of MIDI file paths
        """
        dir_path = Path(directory)
        midi_files = []
        
        for ext in self.config.supported_midi_extensions:
            if recursive:
                midi_files.extend(dir_path.rglob(f"*{ext}"))
            else:
                midi_files.extend(dir_path.glob(f"*{ext}"))
        
        return sorted(midi_files)
    
    def get_fasta_files(self, directory: str, recursive: bool = True) -> List[Path]:
        """
        Get all FASTA files from a directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to search recursively
            
        Returns:
            List of FASTA file paths
        """
        dir_path = Path(directory)
        fasta_files = []
        
        for ext in self.config.supported_fasta_extensions:
            if recursive:
                fasta_files.extend(dir_path.rglob(f"*{ext}"))
            else:
                fasta_files.extend(dir_path.glob(f"*{ext}"))
        
        return sorted(fasta_files)
    
    def load_from_config(self, config_path: str) -> Dict[str, List[str]]:
        """
        Load data directories from pipeline configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            Dictionary with 'midi_dirs' and 'fasta_dirs'
        """
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        result = {'midi_dirs': [], 'fasta_dirs': []}
        
        # Check for explicit data paths in config
        if 'data_paths' in config:
            data_paths = config['data_paths']
            if 'midi' in data_paths:
                result['midi_dirs'] = data_paths['midi']
            if 'fasta' in data_paths:
                result['fasta_dirs'] = data_paths['fasta']
        
        # Also check common locations
        project_root = Path(config_path).parent.parent
        default_midi_dir = project_root / 'data' / 'midi'
        default_fasta_dir = project_root / 'data' / 'fasta'
        
        if default_midi_dir.exists() and not result['midi_dirs']:
            result['midi_dirs'].append(str(default_midi_dir))
        
        if default_fasta_dir.exists() and not result['fasta_dirs']:
            result['fasta_dirs'].append(str(default_fasta_dir))
        
        return result
    
    def create_sample_config(self, output_path: str, 
                            midi_dirs: Optional[List[str]] = None,
                            fasta_dirs: Optional[List[str]] = None):
        """
        Create a sample configuration file for data paths.
        
        Args:
            output_path: Path to save configuration file
            midi_dirs: List of MIDI data directories
            fasta_dirs: List of FASTA data directories
        """
        config = {
            'data_paths': {
                'midi': midi_dirs or [],
                'fasta': fasta_dirs or []
            },
            'loading_options': {
                'min_midi_duration': self.config.min_midi_duration,
                'max_midi_duration': self.config.max_midi_duration,
                'min_sequence_length': self.config.min_sequence_length,
                'recursive_search': True
            }
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Sample configuration saved to: {output_path}")
        return config
    
    def scan_and_report(self, search_path: Optional[str] = None) -> Dict:
        """
        Scan for available data and generate a report.
        
        Args:
            search_path: Starting path for search
            
        Returns:
            Dictionary with scan results
        """
        directories = self.find_data_directories(search_path)
        
        report = {
            'midi_files': {},
            'fasta_files': {},
            'total_midi_count': 0,
            'total_fasta_count': 0
        }
        
        # Count MIDI files
        for dir_path in directories['midi_dirs']:
            files = self.get_midi_files(str(dir_path), recursive=False)
            if files:
                report['midi_files'][str(dir_path)] = len(files)
                report['total_midi_count'] += len(files)
        
        # Count FASTA files
        for dir_path in directories['fasta_dirs']:
            files = self.get_fasta_files(str(dir_path), recursive=False)
            if files:
                report['fasta_files'][str(dir_path)] = len(files)
                report['total_fasta_count'] += len(files)
        
        return report
    
    def print_data_summary(self, search_path: Optional[str] = None):
        """
        Print a summary of available data files.
        
        Args:
            search_path: Starting path for search
        """
        report = self.scan_and_report(search_path)
        
        print("\n" + "=" * 60)
        print("AVAILABLE DATA FILES SUMMARY")
        print("=" * 60)
        
        if report['midi_files']:
            print(f"\nMIDI Files ({report['total_midi_count']} total):")
            for dir_path, count in report['midi_files'].items():
                print(f"  {dir_path}: {count} files")
        else:
            print("\nNo MIDI files found.")
        
        if report['fasta_files']:
            print(f"\nFASTA Files ({report['total_fasta_count']} total):")
            for dir_path, count in report['fasta_files'].items():
                print(f"  {dir_path}: {count} files")
        else:
            print("\nNo FASTA files found.")
        
        print("\n" + "=" * 60)
        
        return report


def setup_user_datasets(project_root: str, 
                       midi_subdir: str = 'data/midi',
                       fasta_subdir: str = 'data/fasta') -> Dict[str, str]:
    """
    Setup standard directories for user datasets.
    
    This function creates standard directory structure for users
    to place their MIDI and FASTA files.
    
    Args:
        project_root: Root directory of the project
        midi_subdir: Subdirectory for MIDI files (relative to project_root)
        fasta_subdir: Subdirectory for FASTA files (relative to project_root)
        
    Returns:
        Dictionary with created directory paths
    """
    project_root = Path(project_root)
    
    midi_dir = project_root / midi_subdir
    fasta_dir = project_root / fasta_subdir
    
    # Create directories
    midi_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir.mkdir(parents=True, exist_ok=True)
    
    # Create README files with instructions
    midi_readme = midi_dir / 'README.txt'
    with open(midi_readme, 'w') as f:
        f.write("""MIDI Dataset Directory
======================

Place your MIDI files (.mid, .midi) in this directory or any subdirectory.
The pipeline will automatically discover and process all MIDI files found here.

Supported formats:
- .mid
- .midi

You can organize files in subdirectories if needed:
- data/midi/classical/
- data/midi/jazz/
- data/midi/custom/

All files will be included in the dataset.
""")
    
    fasta_readme = fasta_dir / 'README.txt'
    with open(fasta_readme, 'w') as f:
        f.write("""FASTA Dataset Directory
=======================

Place your FASTA files (.fasta, .fa, .fna, .ffn, .faa, .frn) in this directory 
or any subdirectory. The pipeline will automatically discover and process 
all FASTA files found here.

Supported formats:
- .fasta
- .fa
- .fna
- .ffn
- .faa
- .frn

You can organize files in subdirectories if needed:
- data/fasta/human/
- data/fasta/microbial/
- data/fasta/custom/

All files will be included in the dataset.
""")
    
    print(f"Created MIDI directory: {midi_dir}")
    print(f"Created FASTA directory: {fasta_dir}")
    
    return {
        'midi_dir': str(midi_dir),
        'fasta_dir': str(fasta_dir)
    }


if __name__ == '__main__':
    # Example usage
    loader = UniversalDataLoader()
    loader.print_data_summary()
