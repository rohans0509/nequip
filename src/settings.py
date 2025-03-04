# src/settings.py
"""
Global settings for the experiment management system.

- BASE_DIR and TEMPLATE_DIR define where results and configuration templates are stored.
- EXPERIMENT_NAME names the experiment.
- PARAM_GRID includes hyperparameters for config generation, now including 'dataset'.
- BASE_CONFIGS is a dictionary mapping each dataset to its corresponding base config file.
- Other training/model parameters (e.g., BATCH_SIZE, TOTAL_LAYERS) are also defined here.
"""

from pathlib import Path
import multiprocessing as mp

# Base directories
BASE_DIR = Path("src/results")
TEMPLATE_DIR = Path("src/config_templates")

# Experiment settings
EXPERIMENT_NAME = "aspirin_e3nn_study"

# Testing flag - set to True for test runs, False for production runs
DEFAULT_IS_TEST = False

# Multiple runs settings
NUM_RUNS = 3  # Number of runs for each configuration
RUN_LABELS = [f"run_{i+1}" for i in range(NUM_RUNS)]  # Labels for each run ("run_1", "run_2", etc.)

# Setting to control whether multiple runs are executed in parallel
# When False, runs are executed sequentially (safer for GPU memory)
PARALLEL_RUNS = False  

# Maximum number of concurrent processes for experiment execution
# Set to a number lower than your available CPU cores to be considerate of other users
MAX_WORKERS = min(4, mp.cpu_count() - 1)

# Model architecture settings
TOTAL_LAYERS = 4  # Total number of layers in the network

# Hyperparameter grid (include dataset as one of the keys)
# PARAM_GRID = {
#     'dataset': ['aspirin'],  # Two datasets
#     'n_train': [100, 200, 400, 800, 900],
#     'lmax': [0, 1, 2, 3, 4, 5],
#     'num_features': [16],
#     'max_epochs': [200],
#     'n_val': [74],
#     'inv_layers': [1, 2, 3, 4],  # For lmax==0, only inv_layers==TOTAL_LAYERS is allowed.
# }

PARAM_GRID = {
    'dataset': ['aspirin', 'benzene', 'ethanol', 'malonaldehyde', 'toluene'],  # Five datasets
    'n_train': [100, 200],
    'lmax': [0, 1, 2],
    'num_features': [16],
    'max_epochs': [10],
    'n_val': [74],
    'inv_layers': [1, 2],  # For lmax==0, only inv_layers==TOTAL_LAYERS is allowed.
}

# Mapping of each dataset to its corresponding base configuration file.
BASE_CONFIGS = {
    'aspirin': TEMPLATE_DIR / "aspirin.yaml",
    'benzene': TEMPLATE_DIR / "benzene.yaml",
    'ethanol': TEMPLATE_DIR / "ethanol.yaml", 
    'malonaldehyde': TEMPLATE_DIR / "malonaldehyde.yaml",
    'toluene': TEMPLATE_DIR / "toluene.yaml",
}

# Training settings
BATCH_SIZE = 50
