# src/settings.py
from pathlib import Path

# Base directories
BASE_DIR = Path("src/results")
TEMPLATE_DIR = Path("src/config_templates")

# Experiment settings
EXPERIMENT_NAME = "aspirin_e3nn_study"

# Model architecture settings
TOTAL_LAYERS = 3  # Added this line

# Model parameters grid
PARAM_GRID = {
    'n_train': [100,200],
    'lmax': [2,4],
    'num_features': [16],
    'max_epochs': [3],
    'n_val': [74],
    'inv_layers': [1,2],
}

# Training settings
BATCH_SIZE = 50
BASE_CONFIG_PATH = TEMPLATE_DIR / "aspirin.yaml"