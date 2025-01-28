# src/managers/visualization_manager.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

# Example logger import (make sure you have this in your codebase)
from src.managers.logging_manager import LoggingManager

class VisualizationManager:
    def __init__(self, experiment_name: str, base_dir: str):
        """
        Visualization Manager that handles plotting experiment metrics.

        Args:
            experiment_name: Name of the experiment (e.g., "aspirin_e3nn_study")
            base_dir: Base directory where experiment folders are located (e.g., "results/")
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        self.version_dir = self._get_latest_version()
        self.plots_dir = self.version_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        self.logger = LoggingManager()

        # Configure Seaborn and Matplotlib for consistent plotting
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [10, 6]

    def _get_latest_version(self) -> Path:
        """
        Get the most recent experiment version directory within base_dir/experiment_name.
        Raises ValueError if no version directories exist.
        """
        experiment_dir = self.base_dir / self.experiment_name
        versions = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if not versions:
            raise ValueError(f"No version directories found in {experiment_dir}")
        # Return the directory with the most recent modification time
        return max(versions, key=lambda x: x.stat().st_mtime)

    def _parse_run_name(self, run_name: str) -> Dict[str, str]:
        """
        Parse run_name into a dictionary of parameters based on keys in src.settings.PARAM_GRID.

        Example run names might look like:
          - "n_train_100_lmax_2_inv_layers_1_num_features_16_max_epochs_3"
          - "default_run" (returns empty dict)
        """
        if run_name == "default_run":
            return {}
        
        # NOTE: Adjust import if your settings file is located elsewhere
        from src.settings import PARAM_GRID
        
        params = {}
        parts = run_name.split('_')
        valid_params = list(PARAM_GRID.keys())

        i = 0
        while i < len(parts):
            matched = False
            for param in valid_params:
                # Split param by underscore if it has multiple words (e.g. "n_train")
                param_parts = param.split('_')
                if i + len(param_parts) <= len(parts):
                    if parts[i:i+len(param_parts)] == param_parts:
                        # The next part in the run name is the parameter value
                        value_index = i + len(param_parts)
                        if value_index < len(parts):
                            params[param] = parts[value_index]
                            i += len(param_parts) + 1
                            matched = True
                            break
            if not matched:
                i += 1
        
        return params

    def _load_metrics(self, run_dir: Path) -> pd.DataFrame:
        """
        Load metrics from a run directory (metrics_epoch.csv).
        Returns an empty DataFrame if not found.
        """
        metrics_path = run_dir / 'metrics_epoch.csv'
        if metrics_path.exists():
            return pd.read_csv(metrics_path, skipinitialspace=True)
        return pd.DataFrame()

    def plot_learning_curves(self, metric: str, epoch: int = -1):
        """
        A 'legacy' style plot: For each run, parse n_train and lmax, 
        then plot the chosen epoch's metric vs. n_train, grouped by lmax.

        NOTE: This only accounts for n_train and lmax; 
        other varying params (inv_layers, etc.) are ignored, 
        which might confound results if they also vary.
        """
        self.logger.section(f"Plotting learning curves for metric='{metric}', epoch={epoch}")
        
        run_dirs = [
            d for d in self.version_dir.iterdir()
            if d.is_dir() and d.name not in
               ['configs', 'logs', 'metrics', 'plots', 'checkpoints']
               and ("processed_dataset" not in d.name)
        ]

        all_data = []
        for run_dir in run_dirs:
            metrics_df = self._load_metrics(run_dir)
            if metrics_df.empty:
                self.logger.warning(f"No metrics found for {run_dir.name}")
                continue
            if metric not in metrics_df.columns:
                self.logger.warning(f"Metric '{metric}' not in {run_dir.name}")
                continue
            
            # Pick the row for the chosen epoch
            if epoch == -1:
                row_idx = -1
            else:
                if epoch >= len(metrics_df):
                    self.logger.warning(f"Epoch {epoch} out of range in {run_dir.name}")
                    continue
                row_idx = epoch

            # Parse run params
            params = self._parse_run_name(run_dir.name)
            if 'n_train' not in params or 'lmax' not in params:
                self.logger.warning(f"Skipping run_dir '{run_dir.name}' because 'n_train' or 'lmax' not found")
                continue
            
            # Build row data
            try:
                row_data = {
                    'n_train': int(params['n_train']),
                    'lmax': int(params['lmax']),
                    'metric_value': metrics_df.iloc[row_idx][metric]
                }
                all_data.append(row_data)
            except ValueError:
                self.logger.warning(f"Could not parse numeric param from {run_dir.name}")
                continue

        if not all_data:
            self.logger.warning("No data to plot for learning curves.")
            return
        
        # Plot
        plot_df = pd.DataFrame(all_data)
        plot_df = plot_df.sort_values("n_train")

        plt.figure(figsize=(10, 6))
        for lmax_val in sorted(plot_df['lmax'].unique()):
            subset = plot_df[plot_df['lmax'] == lmax_val]
            plt.plot(subset['n_train'], subset['metric_value'], marker='o', label=f'lmax={lmax_val}')

        plt.xscale('log')
        plt.xlabel("Number of Training Examples")
        plt.ylabel(metric.replace("_", " ").title())
        plt.title(f"Learning Curves: {metric} (epoch={epoch})")
        plt.legend(title='lmax')
        
        # Save
        save_name = f"learning_curves_{metric}_epoch{epoch}.png"
        save_path = self.plots_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        self.logger.success(f"Plot saved to {save_path}")

    def plot_param_comparison(self,
                              metric: str,
                              epoch: int = -1,
                              fixed_params: Optional[Dict[str, str]] = None):
        """
        Plot how the chosen metric varies with (n_train, lmax, inv_layers) 
        while filtering out runs that do NOT match `fixed_params`.

        We create a FacetGrid with:
          - col="inv_layers"
          - hue="lmax"
          - x="n_train"
          - y="metric_value"

        Args:
            metric: e.g. "validation_loss" or "validation_f_mae"
            epoch: Which epoch index to fetch from each run's metrics. -1 => final epoch
            fixed_params: dict of param_name -> str(value) that must match for all runs used
                          e.g. {'num_features': '16', 'max_epochs': '3'}
                          If not None, runs that don't match exactly will be skipped.
        """
        if fixed_params is None:
            fixed_params = {}

        self.logger.section(f"Plot Param Comparison for metric='{metric}' (epoch={epoch})")

        # 1. Collect run directories
        run_dirs = [
            d for d in self.version_dir.iterdir() 
            if d.is_dir() and d.name not in 
               ['configs', 'logs', 'metrics', 'plots', 'checkpoints'] 
               and ("processed_dataset" not in d.name)
        ]
        if not run_dirs:
            self.logger.warning("No run directories found for plotting.")
            return

        # 2. Parse data
        all_data = []
        for run_dir in run_dirs:
            metrics_df = self._load_metrics(run_dir)
            if metrics_df.empty:
                self.logger.warning(f"No metrics file in {run_dir.name}")
                continue
            if metric not in metrics_df.columns:
                self.logger.warning(f"Metric {metric} not found in {run_dir.name}")
                continue

            # Choose epoch row
            if epoch == -1:
                row_idx = -1
            else:
                if epoch >= len(metrics_df):
                    self.logger.warning(f"Epoch {epoch} out of range in {run_dir.name}")
                    continue
                row_idx = epoch
            
            # Parse run name to extract parameters
            params = self._parse_run_name(run_dir.name)

            # Check if run matches all fixed_params
            skip_run = False
            for key, val in fixed_params.items():
                if key not in params or params[key] != val:
                    skip_run = True
                    break
            if skip_run:
                continue

            # Build row data
            # Convert certain params (n_train, lmax, inv_layers) to int for numeric plotting
            row_data = {
                'metric_value': metrics_df.iloc[row_idx][metric],
            }
            for p_name, p_val in params.items():
                if p_name in ['n_train','lmax','inv_layers']:
                    try:
                        row_data[p_name] = int(p_val)
                    except ValueError:
                        continue
                else:
                    row_data[p_name] = p_val
            all_data.append(row_data)

        if not all_data:
            self.logger.warning("No data left after filtering; nothing to plot.")
            return

        # 3. DataFrame for plotting
        plot_df = pd.DataFrame(all_data)
        
        # Ensure necessary columns exist
        for col in ['n_train', 'lmax', 'inv_layers', 'metric_value']:
            if col not in plot_df.columns:
                self.logger.warning(f"Missing required column: {col}. Cannot plot.")
                return
        
        # 4. Create FacetGrid: columns = inv_layers, hue = lmax, x = n_train
        g = sns.relplot(
            data=plot_df,
            x="n_train",
            y="metric_value",
            hue="lmax",
            col="inv_layers",
            kind="line",
            facet_kws={"sharey": False, "sharex": True}
        )
        g.set(xscale='log')
        g.set_titles(col_template="inv_layers = {col_name}")

        # Optional: a bit of a supertitle
        plt.suptitle(
            f"{metric} vs. n_train\n(lmax colored, inv_layers faceted, epoch={epoch})",
            y=1.02, fontsize=14
        )
        plt.subplots_adjust(top=0.85)  # if suptitle overlaps

        # 5. Save figure
        filename = f"param_comparison_{metric}_epoch{epoch}.png"
        save_path = self.plots_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.success(f"Saved comparison plot to {save_path}")

    def visualize_results(self):
        """
        Example method that shows how you might generate multiple plots
        for your typical metrics. You can adapt or remove as needed.
        """
        self.logger.section("Visualization Suite Generation")

        metrics = [
            'training_loss',
            'validation_loss',
            'training_f_mae',
            'validation_f_mae',
            'training_e_mae',
            'validation_e_mae'
        ]

        self.logger.info(f"Generating plots for {len(metrics)} metrics...")

        # Possibly add a progress bar if you have a logging manager that supports it:
        with self.logger.create_progress() as progress:
            task = progress.add_task("[cyan]Generating plots...", total=len(metrics))
            
            for metric in metrics:
                try:
                    # Example usage of the "legacy" style:
                    self.plot_learning_curves(metric=metric, epoch=-1)
                    
                    # Example usage of the new param comparison approach:
                    # Fix num_features = 16, max_epochs = 3 (as in your settings.py defaults)
                    self.plot_param_comparison(
                        metric=metric,
                        epoch=-1,
                        fixed_params={}
                    )
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate plots for {metric}: {str(e)}")
                
                progress.advance(task)

        self.logger.success("Visualization suite generation completed!")
