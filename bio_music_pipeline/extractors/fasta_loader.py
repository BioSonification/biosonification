"""
FASTA Dataset Loader for user-provided genomic sequences.

This module provides flexible loading of FASTA files from any directory
within the project structure, with support for batch processing and
integration with the bio-vector extraction pipeline.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass


@dataclass
class FastaSequence:
    """Represents a single FASTA sequence."""
    header: str
    sequence: str
    source_file: str
    sequence_id: int = 0
    
    def __post_init__(self):
        """Clean and validate sequence."""
        self.sequence = self.sequence.upper()
        self.sequence = re.sub(r'[^ACGT]', '', self.sequence)
    
    @property
    def length(self) -> int:
        return len(self.sequence)
    
    @property
    def gc_content(self) -> float:
        if len(self.sequence) == 0:
            return 0.0
        gc_count = self.sequence.count('G') + self.sequence.count('C')
        return gc_count / len(self.sequence)


class FastaDatasetLoader:
    """
    Loader for user-provided FASTA datasets.
    
    Supports loading from any directory within the project structure.
    Users can place their FASTA files in any subdirectory.
    """
    
    def __init__(self, 
                 min_sequence_length: int = 100,
                 max_sequences: Optional[int] = None,
                 supported_extensions: Tuple[str, ...] = ('.fasta', '.fa', '.fna', '.ffn', '.faa', '.frn')):
        """
        Initialize FASTA dataset loader.
        
        Args:
            min_sequence_length: Minimum valid sequence length
            max_sequences: Maximum number of sequences to load (None for unlimited)
            supported_extensions: Supported file extensions
        """
        self.min_sequence_length = min_sequence_length
        self.max_sequences = max_sequences
        self.supported_extensions = supported_extensions
        self.project_root = Path(__file__).parent.parent.parent
    
    def read_fasta_file(self, filepath: str) -> Generator[Tuple[str, str], None, None]:
        """
        Read sequences from a single FASTA file.
        
        Args:
            filepath: Path to FASTA file
            
        Yields:
            Tuples of (header, sequence)
        """
        current_header = None
        current_seq = []
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_header is not None:
                            seq = ''.join(current_seq)
                            yield current_header, seq
                        current_header = line[1:]
                        current_seq = []
                    else:
                        current_seq.append(line)
                
                # Don't forget the last sequence
                if current_header is not None:
                    seq = ''.join(current_seq)
                    yield current_header, seq
                    
        except FileNotFoundError:
            raise FileNotFoundError(f"FASTA file not found: {filepath}")
        except Exception as e:
            raise IOError(f"Error reading FASTA file {filepath}: {e}")
    
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
        
        for ext in self.supported_extensions:
            if recursive:
                fasta_files.extend(dir_path.rglob(f"*{ext}"))
            else:
                fasta_files.extend(dir_path.glob(f"*{ext}"))
        
        return sorted(fasta_files)
    
    def load_from_directory(self, 
                           directory: str,
                           recursive: bool = True) -> List[FastaSequence]:
        """
        Load all sequences from a directory.
        
        Args:
            directory: Path to directory containing FASTA files
            recursive: Whether to search recursively
            
        Returns:
            List of FastaSequence objects
        """
        sequences = []
        seq_id = 0
        dir_path = Path(directory)
        
        fasta_files = self.get_fasta_files(str(dir_path), recursive=recursive)
        
        for filepath in fasta_files:
            if self.max_sequences and seq_id >= self.max_sequences:
                break
                
            try:
                for header, sequence in self.read_fasta_file(str(filepath)):
                    # Clean sequence
                    clean_seq = re.sub(r'[^ACGT]', '', sequence.upper())
                    
                    if len(clean_seq) >= self.min_sequence_length:
                        sequences.append(FastaSequence(
                            header=header,
                            sequence=clean_seq,
                            source_file=str(filepath),
                            sequence_id=seq_id
                        ))
                        seq_id += 1
                        
                        if self.max_sequences and seq_id >= self.max_sequences:
                            break
            except Exception as e:
                print(f"Warning: Error processing {filepath}: {e}")
        
        return sequences
    
    def load_from_multiple_directories(self, 
                                       directories: List[str],
                                       recursive: bool = True) -> List[FastaSequence]:
        """
        Load sequences from multiple directories.
        
        Args:
            directories: List of directory paths
            recursive: Whether to search recursively
            
        Returns:
            List of FastaSequence objects
        """
        all_sequences = []
        
        for directory in directories:
            sequences = self.load_from_directory(directory, recursive=recursive)
            all_sequences.extend(sequences)
            
            if self.max_sequences and len(all_sequences) >= self.max_sequences:
                all_sequences = all_sequences[:self.max_sequences]
                break
        
        return all_sequences
    
    def get_statistics(self, sequences: List[FastaSequence]) -> Dict:
        """
        Compute statistics for loaded sequences.
        
        Args:
            sequences: List of FastaSequence objects
            
        Returns:
            Dictionary with statistics
        """
        if not sequences:
            return {'count': 0}
        
        lengths = [seq.length for seq in sequences]
        gc_contents = [seq.gc_content for seq in sequences]
        
        # Count by source file
        source_counts = {}
        for seq in sequences:
            source = Path(seq.source_file).name
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            'count': len(sequences),
            'total_bases': sum(lengths),
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': sum(lengths) / len(lengths),
            'median_length': sorted(lengths)[len(lengths) // 2],
            'min_gc_content': min(gc_contents),
            'max_gc_content': max(gc_contents),
            'mean_gc_content': sum(gc_contents) / len(gc_contents),
            'source_files': source_counts
        }
    
    def print_summary(self, sequences: List[FastaSequence]):
        """
        Print summary of loaded sequences.
        
        Args:
            sequences: List of FastaSequence objects
        """
        stats = self.get_statistics(sequences)
        
        print("\n" + "=" * 60)
        print("FASTA DATASET SUMMARY")
        print("=" * 60)
        print(f"Total sequences: {stats['count']}")
        print(f"Total bases: {stats['total_bases']:,}")
        print(f"\nSequence Length:")
        print(f"  Min: {stats['min_length']:,}")
        print(f"  Max: {stats['max_length']:,}")
        print(f"  Mean: {stats['mean_length']:,.1f}")
        print(f"  Median: {stats['median_length']:,}")
        print(f"\nGC Content:")
        print(f"  Min: {stats['min_gc_content']:.3f}")
        print(f"  Max: {stats['max_gc_content']:.3f}")
        print(f"  Mean: {stats['mean_gc_content']:.3f}")
        
        if stats.get('source_files'):
            print(f"\nSource Files:")
            for filename, count in sorted(stats['source_files'].items()):
                print(f"  {filename}: {count} sequences")
        
        print("=" * 60)


def load_user_fasta_dataset(directories: List[str],
                           min_length: int = 100,
                           max_sequences: Optional[int] = None,
                           verbose: bool = True) -> List[FastaSequence]:
    """
    Convenience function to load user-provided FASTA dataset.
    
    Args:
        directories: List of directory paths containing FASTA files
        min_length: Minimum sequence length
        max_sequences: Maximum number of sequences to load
        verbose: Whether to print summary
        
    Returns:
        List of FastaSequence objects
    """
    loader = FastaDatasetLoader(
        min_sequence_length=min_length,
        max_sequences=max_sequences
    )
    
    sequences = loader.load_from_multiple_directories(directories)
    
    if verbose and sequences:
        loader.print_summary(sequences)
    
    return sequences


if __name__ == '__main__':
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        directories = sys.argv[1:]
    else:
        # Default to project data/fasta directory
        project_root = Path(__file__).parent.parent.parent
        directories = [str(project_root / 'data' / 'fasta')]
    
    sequences = load_user_fasta_dataset(directories)
