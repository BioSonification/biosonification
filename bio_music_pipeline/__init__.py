"""
Bio-Music Pipeline - Main package initialization.
Reproducible controlled symbolic music generation conditioned on biological sequence features.
"""

__version__ = "1.0.0"
__author__ = "Bio-Music Research Team"

import torch
import numpy as np
import random

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
