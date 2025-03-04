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
from src.settings import TOTAL_LAYERS, BASE_DIR, EXPERIMENT_NAME

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
                        row = self._parse_run_directory(run_dir, version)
                        if row:
                            rows.append(row)
        self.df = pd.DataFrame(rows, columns=self.columns)
        self.save_experiment_df()
        return self.df

    def _parse_run_directory(self, run_dir: Path, version: str):
        """
        Parse a single run directory to extract metadata, hyperparameters, metrics, and file paths.
        """
        row = {}
        # Use the run folder name as the experiment_id and run_name.
        row["experiment_id"] = run_dir.name  
        row["experiment_name"] = self.experiment_name
        # Instead of defaulting, extract the dataset from the run name (or config) using our parser.
        try:
            from src.utils.parse_helpers import parse_run_name
            params = parse_run_name(run_dir.name)
        except Exception as e:
            print(f"Error parsing run name {run_dir.name}: {e}")
            params = {}
        row["dataset"] = params.get("dataset", "default")
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

        # Parse hyperparameters from run name
        row["n_train"] = int(params.get("n_train", 0)) if "n_train" in params else None
        row["n_val"] = int(params.get("n_val", 0)) if "n_val" in params else None
        row["lmax"] = int(params.get("lmax", 0)) if "lmax" in params else None
        row["inv_layers"] = int(params.get("inv_layers", 0)) if "inv_layers" in params else None
        row["num_features"] = int(params.get("num_features", 0)) if "num_features" in params else None
        row["max_epochs"] = int(params.get("max_epochs", 0)) if "max_epochs" in params else None

        from src.settings import BATCH_SIZE
        row["batch_size"] = BATCH_SIZE
        row["TOTAL_LAYERS"] = TOTAL_LAYERS

        row["layer_irreps"] = params.get("layer_irreps", "")

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
                if row["max_epochs"] and row["wall_time"]:
                    row["time_per_epoch"] = float(row["wall_time"]) / row["max_epochs"]
                else:
                    row["time_per_epoch"] = None
                if row["n_train"] and row["wall_time"]:
                    row["time_per_sample"] = float(row["wall_time"]) / row["n_train"]
                else:
                    row["time_per_sample"] = None
            except Exception as e:
                print(f"Error loading metrics for {run_dir.name}: {e}")
                row["final_training_loss"] = row["final_validation_loss"] = None
                row["final_training_f_mae"] = row["final_validation_f_mae"] = None
                row["final_training_e_mae"] = row["final_validation_e_mae"] = None
                row["wall_time"] = row["time_per_epoch"] = row["time_per_sample"] = None
        else:
            row["final_training_loss"] = row["final_validation_loss"] = None
            row["final_training_f_mae"] = row["final_validation_f_mae"] = None
            row["final_training_e_mae"] = row["final_validation_e_mae"] = None
            row["wall_time"] = row["time_per_epoch"] = row["time_per_sample"] = None

        # Derived metrics
        row["regression_slope"] = None
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
