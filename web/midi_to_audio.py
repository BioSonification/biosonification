"""
MIDI to Audio Converter

Converts MIDI files to WAV audio format using multiple methods with fallback.
"""

import subprocess
from importlib.util import find_spec
from pathlib import Path
from typing import Optional


def _try_midi2audio(midi_path: str, wav_path: str, soundfont_path: Optional[str] = None) -> bool:
    """
    Try converting MIDI to WAV using midi2audio library.

    Args:
        midi_path: Path to input MIDI file
        wav_path: Path to output WAV file
        soundfont_path: Optional path to soundfont file

    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        from midi2audio import FluidSynth

        # Create output directory if needed
        Path(wav_path).parent.mkdir(parents=True, exist_ok=True)

        # Use provided soundfont or try to find default
        if soundfont_path is None:
            # Try common soundfont locations
            possible_paths = [
                Path(__file__).parent / "static" / "soundfonts" / "default.sf2",
                Path(__file__).parent / "static" / "soundfonts" / "FluidR3_GM.sf2",
                Path(__file__).parent / "static" / "soundfonts" / "FluidR3Mono_GM.sf3",
                Path(__file__).parent / "static" / "soundfonts" / "GeneralUser.sf2",
            ]
            for path in possible_paths:
                if path.exists():
                    soundfont_path = str(path)
                    break

        # Initialize FluidSynth
        fs = FluidSynth(sound_font=soundfont_path)

        # Convert MIDI to WAV
        fs.midi_to_audio(midi_path, wav_path)

        # Check if output file was created
        if Path(wav_path).exists() and Path(wav_path).stat().st_size > 0:
            return True

        return False

    except Exception as e:
        print(f"midi2audio conversion failed: {e}")
        return False


def _try_fluidsynth_cli(midi_path: str, wav_path: str, soundfont_path: Optional[str] = None) -> bool:
    """
    Try converting MIDI to WAV using fluidsynth CLI.

    Args:
        midi_path: Path to input MIDI file
        wav_path: Path to output WAV file
        soundfont_path: Optional path to soundfont file

    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        # Create output directory if needed
        Path(wav_path).parent.mkdir(parents=True, exist_ok=True)

        # Find soundfont if not provided
        if soundfont_path is None:
            possible_paths = [
                Path(__file__).parent / "static" / "soundfonts" / "default.sf2",
                Path(__file__).parent / "static" / "soundfonts" / "FluidR3_GM.sf2",
                Path(__file__).parent / "static" / "soundfonts" / "FluidR3Mono_GM.sf3",
                Path(__file__).parent / "static" / "soundfonts" / "GeneralUser.sf2",
            ]
            for path in possible_paths:
                if path.exists():
                    soundfont_path = str(path)
                    break

        if soundfont_path is None:
            return False

        # Try fluidsynth command
        cmd = [
            "fluidsynth",
            "-ni",  # no interactive mode
            soundfont_path,
            midi_path,
            "-F",
            wav_path,  # output to file
            "-r",
            "44100",  # sample rate
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Check if output file was created
        if result.returncode == 0 and Path(wav_path).exists() and Path(wav_path).stat().st_size > 0:
            return True

        return False

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"fluidsynth CLI conversion failed: {e}")
        return False


def _try_timidity_cli(midi_path: str, wav_path: str) -> bool:
    """
    Try converting MIDI to WAV using timidity CLI.

    Args:
        midi_path: Path to input MIDI file
        wav_path: Path to output WAV file

    Returns:
        True if conversion succeeded, False otherwise
    """
    try:
        # Create output directory if needed
        Path(wav_path).parent.mkdir(parents=True, exist_ok=True)

        # Try timidity command
        cmd = ["timidity", midi_path, "-Ow", "-o", wav_path]  # output WAV

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        # Check if output file was created
        if result.returncode == 0 and Path(wav_path).exists() and Path(wav_path).stat().st_size > 0:
            return True

        return False

    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        print(f"timidity CLI conversion failed: {e}")
        return False


def midi_to_wav(midi_path: str, wav_path: str, soundfont_path: Optional[str] = None) -> bool:
    """
    Convert MIDI to WAV using available methods with fallback.

    Tries methods in order:
    1. midi2audio library (recommended)
    2. fluidsynth CLI
    3. timidity CLI

    Args:
        midi_path: Path to input MIDI file
        wav_path: Path to output WAV file
        soundfont_path: Optional path to soundfont file

    Returns:
        True if conversion succeeded, False otherwise
    """
    # Try midi2audio first (most reliable)
    if _try_midi2audio(midi_path, wav_path, soundfont_path):
        return True

    # Fallback to fluidsynth CLI
    if _try_fluidsynth_cli(midi_path, wav_path, soundfont_path):
        return True

    # Fallback to timidity CLI
    if _try_timidity_cli(midi_path, wav_path):
        return True

    return False


def _check_midi2audio() -> bool:
    """Check if midi2audio library is available."""
    return find_spec("midi2audio") is not None


def _check_fluidsynth_cli() -> bool:
    """Check if fluidsynth CLI is available."""
    try:
        result = subprocess.run(["fluidsynth", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _check_timidity_cli() -> bool:
    """Check if timidity CLI is available."""
    try:
        result = subprocess.run(["timidity", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def check_audio_synthesizer() -> dict:
    """
    Check which audio synthesizers are available.

    Returns:
        Dict with availability status for each method
    """
    return {
        "midi2audio": _check_midi2audio(),
        "fluidsynth": _check_fluidsynth_cli(),
        "timidity": _check_timidity_cli(),
    }


def get_install_instructions() -> str:
    """
    Get installation instructions for audio synthesizers.

    Returns:
        String with installation instructions
    """
    status = check_audio_synthesizer()

    if status["midi2audio"]:
        return "Audio playback enabled via midi2audio library."

    if status["fluidsynth"] or status["timidity"]:
        return "Audio playback enabled via CLI synthesizer."

    return """
Audio playback requires a MIDI synthesizer. Install one of:

1. midi2audio (recommended):
   pip install midi2audio

2. FluidSynth:
   - Windows: choco install fluidsynth
   - macOS: brew install fluid-synth
   - Linux: apt-get install fluidsynth

3. TiMidity++:
   - Download from http://timidity.sourceforge.net/

After installation, restart the web server.
"""
