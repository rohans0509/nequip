# src/settings.py
from pathlib import Path

# Base directories
BASE_DIR = Path("src/results")
TEMPLATE_DIR = Path("src/config_templates")

# Experiment settings
EXPERIMENT_NAME = "aspirin_e3nn_study"

# Model architecture settings
TOTAL_LAYERS = 4  # Added this line

# Model parameters grid
PARAM_GRID = {
    'n_train': [2],
    'lmax': [1],
    'num_features': [32],
    'max_epochs': [2],
    'n_val': [50],
    'inv_layers': [1],
}

# Training settings
BATCH_SIZE = 50
BASE_CONFIG_PATH = TEMPLATE_DIR / "aspirin.yaml"