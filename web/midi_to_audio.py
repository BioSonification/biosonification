"""
MIDI to Audio Converter

Converts MIDI files to WAV audio format.
Currently audio generation is disabled due to fluidsynth CLI compatibility issues.
MIDI files are still available for download and can be played in any DAW or player.
"""

import os
from pathlib import Path
from typing import Optional


def midi_to_wav(midi_path: str, wav_path: str, 
                soundfont_path: Optional[str] = None) -> bool:
    """
    Convert MIDI to WAV. Currently disabled due to fluidsynth CLI compatibility.
    
    Args:
        midi_path: Path to input MIDI file
        wav_path: Path to output WAV file
        soundfont_path: Optional path to soundfont
        
    Returns:
        Always False - audio conversion currently disabled
    """
    # Audio conversion currently disabled due to fluidsynth 2.5.3 CLI changes
    # MIDI files are still available for download
    return False


def check_audio_synthesizer() -> dict:
    """
    Check which audio synthesizers are available.
    
    Returns:
        Dict with availability status (always False currently)
    """
    return {
        'fluidsynth': False,  # Disabled due to CLI compatibility
        'timidity': False,
    }


def get_install_instructions() -> str:
    """
    Get information about audio playback status.
    
    Returns:
        String with status message
    """
    return """
Audio playback is currently disabled due to compatibility issues with fluidsynth 2.5.3.
You can still download MIDI files and play them in:
- GarageBand (macOS)
- MuseScore (free, cross-platform)
- Any DAW or MIDI player
"""
