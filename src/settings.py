# src/settings.py
from pathlib import Path

# Base directories
BASE_DIR = Path("src/results")
TEMPLATE_DIR = Path("src/config_templates")

# Experiment settings
EXPERIMENT_NAME = "aspirin_e3nn_study"

# Model architecture settings
TOTAL_LAYERS = 4  # Added this line

 

# Training settings
BATCH_SIZE = 50
BASE_CONFIG_PATH = TEMPLATE_DIR / "aspirin.yaml"