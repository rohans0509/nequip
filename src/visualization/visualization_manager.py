import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

class VisualizationManager:
    def __init__(self, experiment_name: str, base_dir: str):
        """
        Args:
            experiment_name: Name of the experiment
            base_dir: Base directory for results
        """
        self.base_dir = Path(base_dir)
        self.experiment_name = experiment_name
        self.version_dir = self._get_latest_version()
        self.plots_dir = self.version_dir / 'plots'
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [10, 6]

    def _get_latest_version(self) -> Path:
        """Get the most recent experiment version directory."""
        experiment_dir = self.base_dir / self.experiment_name
        versions = [d for d in experiment_dir.iterdir() if d.is_dir()]
        if not versions:
            raise ValueError(f"No versions found in {experiment_dir}")
        return max(versions, key=lambda x: x.stat().st_mtime)

    def _parse_run_name(self, run_name: str) -> Dict[str, str]:
        """
        Parse run name into parameter dictionary.
        Handles parameters defined in settings.PARAM_GRID.
        
        Example run names:
        - n_train_2_lmax_1
        - batch_size_32_lr_0.001
        - any_param_value_other_param_value
        """
        if run_name == "default_run":
            return {}
            
        params = {}
        parts = run_name.split('_')
        
        # Build parameter names from settings
        from src.settings import PARAM_GRID
        valid_params = PARAM_GRID.keys()
        
        i = 0
        while i < len(parts):
            # Try to find parameter names (they might be single or multi-word)
            for param in valid_params:
                param_parts = param.split('_')
                if i + len(param_parts) <= len(parts):
                    # Check if we found the parameter name
                    if parts[i:i+len(param_parts)] == param_parts:
                        # Get the value (it's right after the parameter name)
                        if i + len(param_parts) < len(parts):
                            params[param] = parts[i + len(param_parts)]
                            i += len(param_parts) + 1
                            break
            else:
                i += 1
        
        print(f"Parsed {run_name} into {params}")  # Debug print
        return params

    def _load_metrics(self, run_dir: Path) -> pd.DataFrame:
        """Load metrics from a run directory."""
        metrics_path = run_dir / 'metrics_epoch.csv'
        if metrics_path.exists():
            return pd.read_csv(metrics_path, skipinitialspace=True)
        return pd.DataFrame()

    def plot_learning_curves(self, metric: str, epoch: int = -1) -> None:
        """
        Plot metric vs n_train for different lmax values.
        
        Args:
            metric: Metric to plot (e.g., 'training_f_mae', 'validation_f_mae')
            epoch: Which epoch to plot (-1 for last epoch)
        """
        # Collect and prepare data
        all_data = []
        for run_dir in self.version_dir.iterdir():
            if (run_dir.is_dir() and 
                not run_dir.name.startswith('processed_dataset_') and
                run_dir.name not in ['configs', 'logs', 'metrics', 'plots', 'checkpoints']):
                
                params = self._parse_run_name(run_dir.name)
                
                # Skip if not a valid run directory
                if 'n_train' not in params or 'lmax' not in params:
                    continue
                    
                metrics_df = self._load_metrics(run_dir)
                if not metrics_df.empty:
                    # Get data for specified epoch
                    epoch_idx = -1 if epoch == -1 else epoch
                    if epoch_idx >= len(metrics_df):
                        print(f"Warning: Epoch {epoch} not found in {run_dir.name}")
                        continue
                        
                    # Create row with parameters and metric value
                    row_data = {
                        'n_train': int(params['n_train']),
                        'lmax': params['lmax'],
                        'metric_value': metrics_df.iloc[epoch_idx][metric]
                    }
                    all_data.append(row_data)
        
        if not all_data:
            print("No matching runs found")
            return

        # Convert to DataFrame
        plot_df = pd.DataFrame(all_data)
        
        # Create plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            data=plot_df,
            x='n_train',
            y='metric_value',
            hue='lmax',
            style='lmax',
            markers=True,
            dashes=False,
            marker='o'
        )

        # Customize plot
        plt.xlabel('Number of Training Examples')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} vs Training Set Size')
        plt.legend(title='lmax')
        plt.grid(True, alpha=0.3)
        
        # Save plot
        plot_name = f'learning_curves_{metric}_epoch{epoch}.png'
        plt.savefig(self.plots_dir / plot_name, dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    def visualize_results(self):
        """Generate all visualization plots for the experiment."""
        # Example metrics to plot
        metrics = ['training_f_mae', 'validation_f_mae']
        parameters = ['inv_layers', 'lmax']
        
        for metric in metrics:
            for fixed_param in parameters:
                for vary_param in parameters:
                    if fixed_param != vary_param:
                        self.plot_learning_curves(
                            metric=metric,
                            fixed_param=fixed_param,
                            vary_param=vary_param
                        )
