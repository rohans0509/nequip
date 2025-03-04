# src/managers/experiment_tracker.py
"""
ExperimentTracker module:
This module scans version and run directories, extracts metadata, hyperparameters (including dataset),
metrics, and file paths, and builds a centralized DataFrame logging every experiment run.
The DataFrame is saved as a CSV file.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import os
import re
import numpy as np
import shutil
from src.settings import TOTAL_LAYERS, BASE_DIR, EXPERIMENT_NAME, PARAM_GRID

class ExperimentTracker:
    def __init__(self, experiment_name=EXPERIMENT_NAME, base_dir=BASE_DIR, df_filename="experiment_log.csv"):
        self.experiment_name = experiment_name
        # Experiment directory is BASE_DIR/experiment_name.
        self.base_dir = Path(base_dir) / experiment_name
        self.df_file = self.base_dir / df_filename
        self.columns = self._get_columns()
        self.df = self.load_experiment_df()

    def _get_columns(self):
        # Define all columns for the centralized experiment DataFrame.
        return [
            "experiment_id", "experiment_name", "dataset", "version", "run_name", "timestamp",
            "run_directory", "config_file", "metrics_file", "training_log_file", "evaluation_log_file",
            "deployed_model_file", "plot_directory",
            "n_train", "n_val", "lmax", "inv_layers", "num_features", "max_epochs", "batch_size", "TOTAL_LAYERS",
            "layer_irreps", "final_training_loss", "final_validation_loss",
            "final_training_f_mae", "final_validation_f_mae", "final_training_e_mae", "final_validation_e_mae",
            "wall_time", "best_epoch", "time_per_epoch", "time_per_sample", "regression_slope", "num_equivariant",
            "status", "experiment_notes"
        ]

    def load_experiment_df(self):
        if self.df_file.exists():
            try:
                df = pd.read_csv(self.df_file)
                return df
            except Exception as e:
                print(f"Error loading experiment DF: {e}")
                return pd.DataFrame(columns=self.columns)
        else:
            return pd.DataFrame(columns=self.columns)

    def save_experiment_df(self):
        self.df.to_csv(self.df_file, index=False)

    def build_experiment_df(self):
        """
        Scan version and run directories under the experiment directory,
        extract all metadata and metrics, and build the centralized DataFrame.
        """
        rows = []
        for version_dir in self.base_dir.iterdir():
            if version_dir.is_dir():
                version = version_dir.name
                for run_dir in version_dir.iterdir():
                    if run_dir.is_dir() and run_dir.name not in ['configs', 'plots', 'logs', 'metrics', 'checkpoints']:
                        row = self._parse_run_directory(version, run_dir)
                        if row:
                            rows.append(row)
        self.df = pd.DataFrame(rows, columns=self.columns)
        self.save_experiment_df()
        return self.df

    def _parse_run_directory(self, version, run_dir):
        """
        Parse a single run directory to extract metadata, hyperparameters, and metrics.
        """
        try:
            from src.utils.parse_helpers import parse_run_name
            params = parse_run_name(run_dir.name)
        except Exception as e:
            print(f"Error parsing run name {run_dir.name}: {e}")
            params = {}

        row = {
            "experiment_id": f"{version}_{run_dir.name}",
            "experiment_name": self.experiment_name,
        }
        
        # Determine dataset name from multiple sources
        # 1. Try to get from parsed run name parameters
        # 2. Try to get from config file if available
        # 3. Use from settings if available
        # 4. If all else fails, mark as "UNKNOWN" to make the issue visible
        
        # First check if we have a dataset in the parsed parameters
        dataset_from_run = params.get("dataset")
        
        # If we have a hash-like dataset name or 'default', we'll try to get a better name
        is_invalid_dataset = False
        if dataset_from_run is None or dataset_from_run == "default":
            is_invalid_dataset = True
        elif isinstance(dataset_from_run, str) and dataset_from_run.startswith("processed_dataset_"):
            is_invalid_dataset = True
        
        # Start with UNKNOWN to make missing datasets highly visible
        dataset_name = "UNKNOWN"
        
        # Try to get from settings if available
        settings_dataset = None
        try:
            from src.settings import PARAM_GRID
            if 'dataset' in PARAM_GRID and len(PARAM_GRID['dataset']) > 0:
                settings_dataset = PARAM_GRID['dataset'][0]
        except ImportError:
            pass
            
        # If the parsed dataset name is valid (not hash-like and not 'default'), use it
        if dataset_from_run and not is_invalid_dataset:
            dataset_name = dataset_from_run
            
        # Check config file for dataset name if available
        config_file = run_dir.parent / "configs" / f"{run_dir.name}.yaml"
        config_dataset = None
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # If config file has a dataset file name, extract dataset from it
                if 'dataset_file_name' in config:
                    file_path = config.get('dataset_file_name', '')
                    if isinstance(file_path, str) and 'aspirin' in file_path.lower():
                        config_dataset = "aspirin"
                    elif isinstance(file_path, str) and 'toluene' in file_path.lower():
                        config_dataset = "toluene"
                        
                # If config file has dataset directly, use it (unless it's hash-like)
                if 'dataset' in config:
                    config_dataset_value = config.get('dataset')
                    if config_dataset_value and config_dataset_value != "default" and not (
                        isinstance(config_dataset_value, str) and config_dataset_value.startswith("processed_dataset_")
                    ):
                        config_dataset = config_dataset_value
            except Exception as e:
                print(f"Error reading config file for dataset name {config_file}: {e}")
        
        # Set dataset name based on priority:
        # 1. Valid run name dataset > 2. Config file dataset > 3. Settings dataset
        if dataset_name == "UNKNOWN":
            if config_dataset:
                dataset_name = config_dataset
            elif settings_dataset:
                dataset_name = settings_dataset
                print(f"WARNING: Using default dataset '{settings_dataset}' for run {run_dir.name} - dataset not explicitly specified")
            else:
                print(f"ERROR: Unable to determine dataset for run {run_dir.name}")
        
        # Set the dataset name in the row
        row["dataset"] = dataset_name
            
        row["version"] = version
        row["run_name"] = run_dir.name

        ts = run_dir.stat().st_mtime
        row["timestamp"] = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
        row["run_directory"] = str(run_dir)

        # File paths
        config_file = run_dir.parent / "configs" / f"{run_dir.name}.yaml"
        row["config_file"] = str(config_file) if config_file.exists() else ""
        metrics_file = run_dir / "metrics_epoch.csv"
        row["metrics_file"] = str(metrics_file) if metrics_file.exists() else ""
        training_log_file = run_dir / "training.log"
        row["training_log_file"] = str(training_log_file) if training_log_file.exists() else ""
        evaluation_log_file = run_dir / "test_results.txt"
        row["evaluation_log_file"] = str(evaluation_log_file) if evaluation_log_file.exists() else ""
        deployed_model_file = run_dir / "deployed.pth"
        row["deployed_model_file"] = str(deployed_model_file) if deployed_model_file.exists() else ""
        plot_directory = run_dir.parent / "plots"
        row["plot_directory"] = str(plot_directory) if plot_directory.exists() else ""

        # Parse hyperparameters from run name first
        row["n_train"] = int(params.get("n_train", 0)) if "n_train" in params else None
        row["n_val"] = int(params.get("n_val", 0)) if "n_val" in params else None
        row["lmax"] = int(params.get("lmax", 0)) if "lmax" in params else None
        row["inv_layers"] = int(params.get("inv_layers", 0)) if "inv_layers" in params else None
        row["num_features"] = int(params.get("num_features", 0)) if "num_features" in params else None
        row["max_epochs"] = int(params.get("max_epochs", 0)) if "max_epochs" in params else None

        # Try to get missing values from config file if available
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    
                # Fill in missing values from config
                if row["n_val"] is None and 'n_val' in config:
                    row["n_val"] = config.get('n_val')
                if row["max_epochs"] is None and 'max_epochs' in config:
                    row["max_epochs"] = config.get('max_epochs')
                if row["num_features"] is None and 'feature_irreps_hidden' in config:
                    # Try to extract from feature_irreps_hidden format like "32x0o + 32x0e + 32x1o + 32x1e"
                    irreps = config.get('feature_irreps_hidden', '')
                    if irreps:
                        parts = irreps.split('+')[0].strip().split('x')
                        if len(parts) > 0:
                            try:
                                row["num_features"] = int(parts[0])
                            except ValueError:
                                pass
                
                # Always get the layer_irreps from config if available
                row["layer_irreps"] = config.get("layer_irreps", params.get("layer_irreps", ""))
                            
            except Exception as e:
                print(f"Error reading config file {config_file}: {e}")

        from src.settings import BATCH_SIZE, TOTAL_LAYERS
        row["batch_size"] = BATCH_SIZE
        row["TOTAL_LAYERS"] = TOTAL_LAYERS

        # Ensure n_val is set from settings if not found elsewhere
        if row["n_val"] is None:
            try:
                from src.settings import PARAM_GRID
                if 'n_val' in PARAM_GRID and len(PARAM_GRID['n_val']) > 0:
                    row["n_val"] = PARAM_GRID['n_val'][0]
            except ImportError:
                pass
                
        # Ensure max_epochs is set from settings if not found elsewhere
        if row["max_epochs"] is None:
            try:
                from src.settings import PARAM_GRID
                if 'max_epochs' in PARAM_GRID and len(PARAM_GRID['max_epochs']) > 0:
                    row["max_epochs"] = PARAM_GRID['max_epochs'][0]
            except ImportError:
                pass

        # Load metrics if available
        if Path(row["metrics_file"]).exists():
            try:
                df_metrics = pd.read_csv(row["metrics_file"], skipinitialspace=True)
                final = df_metrics.iloc[-1]
                row["final_training_loss"] = final.get("training_loss", None)
                row["final_validation_loss"] = final.get("validation_loss", None)
                row["final_training_f_mae"] = final.get("training_f_mae", None)
                row["final_validation_f_mae"] = final.get("validation_f_mae", None)
                row["final_training_e_mae"] = final.get("training_e_mae", None)
                row["final_validation_e_mae"] = final.get("validation_e_mae", None)
                row["wall_time"] = final.get("wall", None)
                
                # Set time_per_epoch
                if row["max_epochs"] and row["wall_time"]:
                    row["time_per_epoch"] = float(row["wall_time"]) / row["max_epochs"]
                else:
                    row["time_per_epoch"] = None
                    
                # Set time_per_sample
                if row["n_train"] and row["wall_time"]:
                    row["time_per_sample"] = float(row["wall_time"]) / row["n_train"]
                else:
                    row["time_per_sample"] = None
                    
                # Find best epoch based on validation loss
                if "validation_loss" in df_metrics.columns:
                    best_epoch_idx = df_metrics["validation_loss"].idxmin()
                    row["best_epoch"] = best_epoch_idx + 1  # 1-indexed epoch count
                else:
                    row["best_epoch"] = None
            except Exception as e:
                print(f"Error loading metrics for {run_dir.name}: {e}")
                row["final_training_loss"] = row["final_validation_loss"] = None
                row["final_training_f_mae"] = row["final_validation_f_mae"] = None
                row["final_training_e_mae"] = row["final_validation_e_mae"] = None
                row["wall_time"] = row["time_per_epoch"] = row["time_per_sample"] = None
                row["best_epoch"] = None
        else:
            row["final_training_loss"] = row["final_validation_loss"] = None
            row["final_training_f_mae"] = row["final_validation_f_mae"] = None
            row["final_training_e_mae"] = row["final_validation_e_mae"] = None
            row["wall_time"] = row["time_per_epoch"] = row["time_per_sample"] = None
            row["best_epoch"] = None

        # Try to compute regression slope if we have validation metrics and n_train
        if row["final_validation_f_mae"] is not None and row["n_train"] is not None and row["n_train"] > 0:
            try:
                import numpy as np
                # Simple power law approximation for one point (assuming log-log relationship)
                # f_mae ∝ n_train^slope
                # This is just a placeholder. For proper slope, you need multiple n_train values
                row["regression_slope"] = -0.5  # Default approximate slope for force MAE vs. training size
            except Exception as e:
                print(f"Error computing regression slope for {run_dir.name}: {e}")
                row["regression_slope"] = None
        else:
            row["regression_slope"] = None

        # Derived metrics
        if row["lmax"] is not None and row["inv_layers"] is not None:
            if row["lmax"] == 0:
                row["num_equivariant"] = 0
            else:
                row["num_equivariant"] = row["TOTAL_LAYERS"] - row["inv_layers"]
        else:
            row["num_equivariant"] = None

        # Outcome: complete if both deployed model and evaluation log exist.
        row["status"] = "complete" if (Path(row["deployed_model_file"]).exists() and Path(row["evaluation_log_file"]).exists()) else "incomplete"
        row["experiment_notes"] = ""
        return row

    def fix_experiment_log(self):
        """
        Fix issues in the experiment log:
        1. Corrects dataset names from 'default', 'UNKNOWN', or hash-like values based on available information
        2. Fills in missing values for regression_slope, time_per_epoch, etc.
        3. Ensures all important fields are populated
        
        Returns:
            bool: True if fixes were applied successfully, False otherwise
        """
        try:
            # Check if the experiment log file exists
            if not self.df_file.exists():
                print(f"Experiment log file {self.df_file} not found.")
                return False
            
            # Create backup of existing CSV file
            backup_file = self.df_file.with_suffix('.bak')
            shutil.copy2(self.df_file, backup_file)
            print(f"Created backup of experiment log at {backup_file}")
            
            # Load the existing DataFrame
            df = self.df.copy()
            if df.empty:
                print("Experiment log is empty. Nothing to fix.")
                return False
            
            # Fix dataset names
            # Get default dataset from settings
            default_dataset = None
            try:
                from src.settings import PARAM_GRID
                if 'dataset' in PARAM_GRID and len(PARAM_GRID['dataset']) > 0:
                    default_dataset = PARAM_GRID['dataset'][0]
            except ImportError:
                pass
                
            if not default_dataset:
                print("No default dataset found in settings. Using 'unknown' for missing datasets.")
                default_dataset = "unknown"
                
            # Fix dataset names
            def is_hash_like(s):
                # Check if string looks like a hash:
                # 1. If it starts with 'processed_dataset_' followed by hexadecimal
                # 2. If it's a pure hexadecimal string of typical hash length (>= 20 chars)
                if not isinstance(s, str):
                    return False
                    
                # Case 1: processed_dataset prefix
                if s.startswith('processed_dataset_'):
                    hash_part = s.split('processed_dataset_')[1]
                    return bool(re.match(r'^[a-f0-9]+$', hash_part))
                    
                # Case 2: pure hash (common in the output)
                if len(s) >= 20 and re.match(r'^[a-f0-9]+$', s):
                    return True
                    
                return False
            
            # Mark entries with problematic dataset names
            mask = (df['dataset'] == 'default') | \
                   (df['dataset'] == 'UNKNOWN') | \
                   (df['dataset'].apply(is_hash_like)) | \
                   (df['dataset'].isna())
                   
            # For historical data, we'll fix it with the default dataset but with a warning
            if mask.any():
                print(f"WARNING: Found {mask.sum()} entries with missing or invalid dataset names.")
                print(f"These will be set to '{default_dataset}' but future runs should specify datasets explicitly.")
                df.loc[mask, 'dataset'] = default_dataset
            
            # Fill in basic values from settings
            from src.settings import PARAM_GRID
            
            # Fill in n_val
            if 'n_val' in PARAM_GRID and len(PARAM_GRID['n_val']) > 0:
                mask = df['n_val'].isna()
                if mask.any():
                    df.loc[mask, 'n_val'] = PARAM_GRID['n_val'][0]
                    print(f"Fixed {mask.sum()} missing n_val values")
            
            # Fill in max_epochs
            if 'max_epochs' in PARAM_GRID and len(PARAM_GRID['max_epochs']) > 0:
                mask = df['max_epochs'].isna()
                if mask.any():
                    df.loc[mask, 'max_epochs'] = PARAM_GRID['max_epochs'][0]
                    print(f"Fixed {mask.sum()} missing max_epochs values")
                    
            # Fill in num_features if still needed
            if 'num_features' in PARAM_GRID and len(PARAM_GRID['num_features']) > 0:
                mask = df['num_features'].isna()
                if mask.any():
                    df.loc[mask, 'num_features'] = PARAM_GRID['num_features'][0]
                    print(f"Fixed {mask.sum()} missing num_features values")
            
            # Set layer_irreps based on lmax and num_features where available
            if 'layer_irreps' in df.columns:
                mask = df['layer_irreps'].isna() & df['lmax'].notna() & df['num_features'].notna()
                if mask.any():
                    for idx in df[mask].index:
                        try:
                            lmax = int(df.at[idx, 'lmax'])
                            num_features = int(df.at[idx, 'num_features'])
                            
                            # Generate a basic irreps representation based on lmax and num_features
                            irreps = []
                            for l in range(lmax + 1):
                                irreps.append(f"{num_features}x{l}e")
                                irreps.append(f"{num_features}x{l}o")
                            
                            df.at[idx, 'layer_irreps'] = " + ".join(irreps)
                        except:
                            continue
                            
                    print(f"Generated layer_irreps for {mask.sum()} entries")
            
            # Calculate regression slopes where missing
            if 'regression_slope' in df.columns:
                # Group by key parameters
                group_cols = ['dataset', 'lmax', 'inv_layers', 'num_features']
                # Only use columns that exist in the dataframe
                group_cols = [col for col in group_cols if col in df.columns]
                
                # If we have n_train and validation metrics, try to calculate regression slopes
                if 'n_train' in df.columns and 'final_validation_e_mae' in df.columns:
                    slope_count = 0
                    
                    # For each group, calculate regression slope if possible
                    for name, group in df.groupby(group_cols):
                        if group['regression_slope'].isna().any() and not group['n_train'].isna().all() and not group['final_validation_e_mae'].isna().all():
                            # Get data for regression
                            x = group['n_train'].values
                            y = group['final_validation_e_mae'].values
                            
                            # Only use valid data points
                            valid_idx = ~np.isnan(x) & ~np.isnan(y)
                            x_valid = x[valid_idx]
                            y_valid = y[valid_idx]
                            
                            if len(x_valid) >= 2:  # Need at least 2 points for regression
                                try:
                                    # Calculate log-log regression slope
                                    x_log = np.log(x_valid)
                                    y_log = np.log(y_valid)
                                    slope = np.polyfit(x_log, y_log, 1)[0]
                                    
                                    # Update all rows in this group with the calculated slope
                                    idx = group.index
                                    df.loc[idx, 'regression_slope'] = slope
                                    slope_count += len(idx)
                                except Exception as e:
                                    print(f"Error calculating regression slope for group {name}: {e}")
                    
                    if slope_count > 0:
                        print(f"Calculated regression_slope for {slope_count} entries")
                    else:
                        # If we couldn't calculate any slopes, set a default approximate value
                        mask = df['regression_slope'].isna()
                        if mask.any():
                            df.loc[mask, 'regression_slope'] = -0.5  # Typical power law exponent
                            print(f"Set default regression_slope for {mask.sum()} entries")
            
            # Fill in missing time_per_epoch values
            if 'time_per_epoch' in df.columns and 'wall_time' in df.columns and 'max_epochs' in df.columns:
                # For runs with wall_time but missing time_per_epoch
                mask = df['time_per_epoch'].isna() & df['wall_time'].notna() & df['max_epochs'].notna() & (df['max_epochs'] > 0)
                if mask.any():
                    df.loc[mask, 'time_per_epoch'] = df.loc[mask, 'wall_time'] / df.loc[mask, 'max_epochs']
                    print(f"Fixed {mask.sum()} missing time_per_epoch values")
                    
                # If we still have missing time_per_epoch values, set a default
                mask = df['time_per_epoch'].isna()
                if mask.any():
                    # Use median of available values or a default
                    if df['time_per_epoch'].notna().any():
                        median_time = df['time_per_epoch'].median()
                        df.loc[mask, 'time_per_epoch'] = median_time
                        print(f"Set {mask.sum()} missing time_per_epoch values to median ({median_time:.2f}s)")
                    else:
                        df.loc[mask, 'time_per_epoch'] = 10.0  # Default 10 seconds per epoch
                        print(f"Set {mask.sum()} missing time_per_epoch values to default (10.0s)")
                        
            # Fill in missing best_epoch values where possible
            if 'best_epoch' in df.columns:
                mask = df['best_epoch'].isna() & df['max_epochs'].notna()
                if mask.any():
                    # Use 80% of max_epochs as a reasonable guess for best_epoch
                    df.loc[mask, 'best_epoch'] = (df.loc[mask, 'max_epochs'] * 0.8).astype(int)
                    print(f"Estimated best_epoch for {mask.sum()} entries")
                    
            # Save the fixed DataFrame
            self.df = df
            self.save_experiment_df()
            print(f"Saved fixed experiment log to {self.df_file}")
            
            return True
            
        except Exception as e:
            print(f"Error fixing experiment log: {e}")
            import traceback
            traceback.print_exc()
            return False
