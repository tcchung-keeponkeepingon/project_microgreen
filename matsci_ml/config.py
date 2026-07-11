"""Configuration constants for materials science ML project."""

# Feature column definitions
COMPOSITIONAL_COLS = ['GUM', 'ALG', 'PVA', 'CMC']
SCALAR_COLS = ['MassLoading']
FEATURE_COLS = COMPOSITIONAL_COLS + SCALAR_COLS

# Default paths
DATA_DIR = './data'
MODEL_DIR = './models'


import random
import numpy as np
import torch
import os

def set_all_seeds(seed=42):
    """Set all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
