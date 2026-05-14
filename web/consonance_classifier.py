"""Consonance classifier for generated MIDI files using CRNN model."""

import sys
import os
from pathlib import Path

# Add diploma path to sys.path to import crnn_music_classifier modules
# Use absolute path that works regardless of how the module is imported
try:
    # Try using __file__ first (most reliable)
    DIPLOMA_ROOT = Path(__file__).resolve().parent.parent.parent / "diploma"
except (NameError, TypeError):
    # Fallback to known absolute path
    DIPLOMA_ROOT = Path("/Users/aloha_kuino/Desktop/diploma")

if str(DIPLOMA_ROOT) not in sys.path:
    sys.path.insert(0, str(DIPLOMA_ROOT))


class ConsonanceClassifier:
    """Wrapper for CRNN consonance classifier."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self._model = None
        self._device = None
        self._is_loaded = False
        self._error = None
        self._torch = None
        self._np = None
        self._crnn_module = None
        self._config = None
        self._features = None

        # Try to resolve model path
        if model_path is None:
            try:
                from crnn_music_classifier.configs.config import BEST_MODEL_PATH
                model_path = BEST_MODEL_PATH
            except ImportError:
                self._error = "Cannot import crnn_music_classifier. Ensure diploma directory is in Python path."
                return

        self.model_path = model_path
        if not Path(model_path).exists():
            self._error = f"Model not found at {model_path}"

    def _load_dependencies(self) -> bool:
        """Lazy-load PyTorch and other dependencies."""
        if self._torch is not None:
            return True

        if self._error:
            return False

        try:
            import torch
            import numpy as np

            # Add crnn_music_classifier directory to path so relative imports work
            crnn_dir = DIPLOMA_ROOT / "crnn_music_classifier"
            if str(crnn_dir) not in sys.path:
                sys.path.insert(0, str(crnn_dir))

            # Import from diploma/crnn_music_classifier
            from configs.config import BEST_MODEL_PATH, CLASS_NAMES
            from model import CRNN
            from utils.features import midi_path_to_melspec

            self._torch = torch
            self._np = np
            self._crnn_module = CRNN
            self._config = (BEST_MODEL_PATH, CLASS_NAMES)
            self._features = midi_path_to_melspec
            return True
        except ImportError as e:
            self._error = f"Failed to import dependencies: {str(e)}"
            return False

    def _load_model(self) -> bool:
        """Lazy-load the model on first use."""
        if self._is_loaded:
            return True

        if not self._load_dependencies():
            return False

        try:
            self._device = self._torch.device("mps" if self._torch.backends.mps.is_available() else "cpu")
            self._model = self._crnn_module()
            self._model.load_state_dict(
                self._torch.load(self.model_path, map_location=self._device, weights_only=True)
            )
            self._model.to(self._device)
            self._model.eval()
            self._is_loaded = True
            return True
        except Exception as e:
            self._error = f"Failed to load model: {str(e)}"
            return False

    def classify(self, midi_path: str) -> dict:
        """Classify MIDI file as consonant or dissonant.

        Returns:
            {
                "success": bool,
                "prediction": str (e.g., "consonant" or "dissonant"),
                "confidence": float (0.0-1.0),
                "scores": {"consonant": float, "dissonant": float},
                "error": str or None
            }
        """
        if not Path(midi_path).exists():
            return {
                "success": False,
                "prediction": None,
                "confidence": None,
                "scores": {},
                "error": f"MIDI file not found: {midi_path}",
            }

        if not self._load_model():
            return {
                "success": False,
                "prediction": None,
                "confidence": None,
                "scores": {},
                "error": self._error,
            }

        try:
            # Extract features from MIDI
            track_specs, mix_spec = self._features(midi_path)

            # Prepare tensors
            track_tensor = (
                self._torch.from_numpy(self._np.stack(track_specs, axis=0))
                .unsqueeze(0)
                .float()
                .to(self._device)
            )
            mix_tensor = self._torch.from_numpy(mix_spec).unsqueeze(0).float().to(self._device)
            num_tracks = len(track_specs)
            track_mask = self._torch.ones(1, num_tracks, dtype=self._torch.bool, device=self._device)

            # Run inference
            with self._torch.no_grad():
                logits = self._model(track_tensor, mix_tensor, track_mask)
                probs = self._torch.softmax(logits, dim=1).cpu().numpy()[0]

            # Get prediction
            _, CLASS_NAMES = self._config
            predicted_idx = int(self._np.argmax(probs))
            prediction = CLASS_NAMES[predicted_idx]
            confidence = float(probs[predicted_idx])

            # Build confidence dict for all classes
            scores = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

            return {
                "success": True,
                "prediction": prediction,
                "confidence": confidence,
                "scores": scores,
                "num_tracks": num_tracks,
                "error": None,
            }

        except Exception as e:
            return {
                "success": False,
                "prediction": None,
                "confidence": None,
                "scores": {},
                "error": f"Classification failed: {str(e)}",
            }

    def is_ready(self) -> bool:
        """Check if classifier is ready to use."""
        return self._error is None

    def get_error(self) -> str:
        """Get initialization error if any."""
        return self._error or ""


# Global classifier instance
_classifier = None


def get_classifier() -> ConsonanceClassifier:
    """Get or create global classifier instance."""
    global _classifier
    if _classifier is None:
        _classifier = ConsonanceClassifier()
    return _classifier
