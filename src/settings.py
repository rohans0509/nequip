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
    'n_train': [100,200,400,800,900],
    'lmax': [1,2,3,4,5],
    'num_features': [16],
    'max_epochs': [200],
    'n_val': [74],
    'inv_layers': [1,2,3],
}

# # Model parameters grid
# PARAM_GRID = {
#     'n_train': [100,200],
#     'lmax': [1,2],
#     'num_features': [16],
#     'max_epochs': [10],
#     'n_val': [74],
#     'inv_layers': [1,2],
# }



# Training settings
BATCH_SIZE = 50
BASE_CONFIG_PATH = TEMPLATE_DIR / "aspirin.yaml"