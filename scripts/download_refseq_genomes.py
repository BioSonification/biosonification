#!/usr/bin/env python3
"""
Download reference genomes from NCBI RefSeq for biosonification training.

This script downloads complete genome sequences for model organisms to expand
the biological dataset from 12 sequences to 1000+ fragments.
"""

import argparse
import gzip
import shutil
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Reference genome URLs from NCBI RefSeq (GCF assemblies)
REFERENCE_GENOMES = {
    "ecoli": {
        "name": "Escherichia coli K-12 MG1655",
        "accession": "GCF_000005845.2",
        "url": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/005/845/GCF_000005845.2_ASM584v2/GCF_000005845.2_ASM584v2_genomic.fna.gz",
        "size_mb": 4.6,
        "description": "Model bacterium, ~4.6 Mbp genome",
    },
    "yeast": {
        "name": "Saccharomyces cerevisiae S288C",
        "accession": "GCF_000146045.2",
        "url": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/146/045/GCF_000146045.2_R64/GCF_000146045.2_R64_genomic.fna.gz",
        "size_mb": 12.2,
        "description": "Baker's yeast, ~12 Mbp genome, 16 chromosomes",
    },
    "fly": {
        "name": "Drosophila melanogaster",
        "accession": "GCF_000001215.4",
        "url": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/215/GCF_000001215.4_Release_6_plus_ISO1_MT/GCF_000001215.4_Release_6_plus_ISO1_MT_genomic.fna.gz",
        "size_mb": 143,
        "description": "Fruit fly, ~143 Mbp genome",
    },
    "worm": {
        "name": "Caenorhabditis elegans",
        "accession": "GCF_000002985.6",
        "url": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/002/985/GCF_000002985.6_WBcel235/GCF_000002985.6_WBcel235_genomic.fna.gz",
        "size_mb": 100,
        "description": "Nematode worm, ~100 Mbp genome",
    },
    "arabidopsis": {
        "name": "Arabidopsis thaliana",
        "accession": "GCF_000001735.4",
        "url": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/735/GCF_000001735.4_TAIR10.1/GCF_000001735.4_TAIR10.1_genomic.fna.gz",
        "size_mb": 119,
        "description": "Model plant, ~119 Mbp genome",
    },
    "human_chr22": {
        "name": "Homo sapiens chromosome 22",
        "accession": "NC_000022.11",
        "url": "https://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.40_GRCh38.p14/GCF_000001405.40_GRCh38.p14_genomic.fna.gz",
        "size_mb": 50,
        "description": "Human chr22 (smallest autosome), ~50 Mbp",
        "extract_chr": "NC_000022.11",
    },
}


def download_file(url: str, output_path: Path, description: str) -> bool:
    """Download a file with progress indication."""
    try:
        print(f"Downloading {description}...")
        print(f"  URL: {url}")

        # Add user agent to avoid 403 errors
        headers = {"User-Agent": "Mozilla/5.0 (biosonification genome downloader)"}
        request = Request(url, headers=headers)

        with urlopen(request, timeout=30) as response:
            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192
            downloaded = 0

            with open(output_path, "wb") as f:
                while True:
                    chunk = response.read(block_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="")

        print(f"\n  [OK] Downloaded to {output_path}")
        return True

    except (URLError, HTTPError) as e:
        print(f"\n  [ERROR] Error downloading: {e}")
        return False
    except Exception as e:
        print(f"\n  [ERROR] Unexpected error: {e}")
        return False


def extract_gz(gz_path: Path, output_path: Path, extract_chr: str = None) -> bool:
    """Extract .gz file, optionally filtering for specific chromosome."""
    try:
        print(f"Extracting {gz_path.name}...")

        with gzip.open(gz_path, "rt") as f_in:
            with open(output_path, "w") as f_out:
                if extract_chr:
                    # Extract only specific chromosome
                    write_mode = False
                    for line in f_in:
                        if line.startswith(">"):
                            # Check if this is the chromosome we want
                            write_mode = extract_chr in line
                        if write_mode:
                            f_out.write(line)
                else:
                    # Copy entire file
                    shutil.copyfileobj(f_in, f_out)

        print(f"  [OK] Extracted to {output_path}")
        return True

    except Exception as e:
        print(f"  [ERROR] Error extracting: {e}")
        return False


def download_genome(genome_key: str, output_dir: Path, keep_gz: bool = False) -> bool:
    """Download and extract a single genome."""
    genome = REFERENCE_GENOMES[genome_key]

    print(f"\n{'='*70}")
    print(f"Genome: {genome['name']}")
    print(f"Accession: {genome['accession']}")
    print(f"Description: {genome['description']}")
    print(f"Expected size: ~{genome['size_mb']} MB")
    print(f"{'='*70}")

    # Prepare file paths
    gz_filename = f"{genome['accession']}_genomic.fna.gz"
    fasta_filename = f"{genome['accession']}_genomic.fna"

    gz_path = output_dir / gz_filename
    fasta_path = output_dir / fasta_filename

    # Skip if already exists
    if fasta_path.exists():
        print(f"  [SKIP] Already exists: {fasta_path}")
        return True

    # Download
    if not download_file(genome["url"], gz_path, genome["name"]):
        return False

    # Extract
    extract_chr = genome.get("extract_chr")
    if not extract_gz(gz_path, fasta_path, extract_chr):
        return False

    # Clean up .gz file unless requested to keep
    if not keep_gz and gz_path.exists():
        gz_path.unlink()
        print(f"  [OK] Removed temporary file {gz_filename}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download reference genomes from NCBI RefSeq",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available organisms:
  ecoli         - Escherichia coli K-12 (~4.6 MB)
  yeast         - Saccharomyces cerevisiae (~12 MB)
  fly           - Drosophila melanogaster (~143 MB)
  worm          - Caenorhabditis elegans (~100 MB)
  arabidopsis   - Arabidopsis thaliana (~119 MB)
  human_chr22   - Homo sapiens chromosome 22 (~50 MB)
  all           - Download all genomes

Examples:
  python download_refseq_genomes.py --organisms ecoli yeast
  python download_refseq_genomes.py --organisms all
  python download_refseq_genomes.py --organisms fly --output-dir custom_dir
        """,
    )

    parser.add_argument(
        "--organisms",
        nargs="+",
        required=True,
        choices=list(REFERENCE_GENOMES.keys()) + ["all"],
        help="Organisms to download",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/fasta/refseq_genomes"),
        help="Output directory for downloaded genomes (default: data/fasta/refseq_genomes)",
    )

    parser.add_argument("--keep-gz", action="store_true", help="Keep compressed .gz files after extraction")

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.output_dir.absolute()}")

    # Determine which genomes to download
    if "all" in args.organisms:
        genomes_to_download = list(REFERENCE_GENOMES.keys())
    else:
        genomes_to_download = args.organisms

    print(f"\nWill download {len(genomes_to_download)} genome(s)")

    # Download each genome
    success_count = 0
    fail_count = 0

    for genome_key in genomes_to_download:
        try:
            if download_genome(genome_key, args.output_dir, args.keep_gz):
                success_count += 1
            else:
                fail_count += 1
        except KeyboardInterrupt:
            print("\n\n[WARNING] Download interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Unexpected error processing {genome_key}: {e}")
            fail_count += 1

        # Small delay between downloads to be nice to NCBI servers
        if genome_key != genomes_to_download[-1]:
            time.sleep(2)

    # Summary
    print(f"\n{'='*70}")
    print(f"DOWNLOAD SUMMARY")
    print(f"{'='*70}")
    print(f"  [OK] Successful: {success_count}")
    print(f"  [ERROR] Failed: {fail_count}")
    print(f"  Output directory: {args.output_dir.absolute()}")

    # Estimate fragments
    if success_count > 0:
        print(f"\nEstimated fragments (with fragment_length=1800, stride=900):")
        total_fragments = 0
        for genome_key in genomes_to_download:
            genome = REFERENCE_GENOMES[genome_key]
            size_bp = genome["size_mb"] * 1_000_000
            fragments = int((size_bp - 1800) / 900) + 1
            total_fragments += fragments
            print(f"  {genome['name']}: ~{fragments:,} fragments")
        print(f"  TOTAL: ~{total_fragments:,} fragments")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
