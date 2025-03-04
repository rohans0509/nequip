# src/utils/parse_helpers.py
"""
Parse Helpers module:
This module provides a helper function to parse experiment run names into a dictionary
of hyperparameters. The valid keys include dataset, n_train, lmax, inv_layers, num_features, max_epochs, and n_val.
"""

def parse_run_name(run_name: str) -> dict:
    """
    Parse a run name string into a dictionary of hyperparameters.
    
    Example:
      "dataset_ds1_n_train_100_lmax_2_inv_layers_1" → 
           {"dataset": "ds1", "n_train": "100", "lmax": "2", "inv_layers": "1"}
    """
    if run_name == "default_run":
        return {}
    parts = run_name.split('_')
    params = {}
    i = 0
    valid_params = ["dataset", "n_train", "lmax", "inv_layers", "num_features", "max_epochs", "n_val"]
    while i < len(parts):
        matched = False
        for key in valid_params:
            key_parts = key.split('_')
            if parts[i:i+len(key_parts)] == key_parts and i+len(key_parts) < len(parts):
                params[key] = parts[i+len(key_parts)]
                i += len(key_parts) + 1
                matched = True
                break
        if not matched:
            i += 1
    return params
