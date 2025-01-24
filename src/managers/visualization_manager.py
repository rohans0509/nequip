import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from src.managers.logging_manager import LoggingManager

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
        self.logger = LoggingManager()
        
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

    def plot_learning_curves(self, metric: str, epoch: int = -1, progress=None) -> None:
        """Plot metric vs n_train for different lmax values."""
        # Collect and prepare data
        all_data = []
        
        # Get all run directories (excluding special folders)
        run_dirs = [d for d in self.version_dir.iterdir() 
                    if d.is_dir() and d.name not in ['configs', 'logs', 'metrics', 'plots', 'checkpoints']]
        
        if not run_dirs:
            self.logger.warning(f"No run directories found in: [highlight]{self.version_dir}[/]")
            return
        
        self.logger.info(f"Processing [highlight]{len(run_dirs)}[/] runs for metric [highlight]{metric}[/]")
        
        # Use existing progress bar if provided, otherwise create new one
        if progress:
            task = progress.add_task(f"[cyan]Processing {metric}...", total=len(run_dirs))
        else:
            progress = self.logger.create_progress()
            task = progress.add_task("[cyan]Processing metrics...", total=len(run_dirs))
            progress.start()
        
        try:
            for run_dir in run_dirs:
                try:
                    metrics_file = run_dir / 'metrics_epoch.csv'
                    if not metrics_file.exists():
                        self.logger.warning(f"[muted]No metrics file in: {run_dir.name}[/]")
                        continue
                        
                    metrics_df = pd.read_csv(metrics_file)
                    
                    # Check if metric exists in file
                    if metric not in metrics_df.columns:
                        self.logger.warning(f"[muted]Metric {metric} not found in {run_dir.name}[/]")
                        continue
                    
                    params = self._parse_run_name(run_dir.name)
                    
                    # Skip if not a valid run directory
                    if 'n_train' not in params or 'lmax' not in params:
                        self.logger.warning(f"[muted]Could not parse parameters from: {run_dir.name}[/]")
                        continue
                        
                    # Get data for specified epoch
                    epoch_idx = -1 if epoch == -1 else epoch
                    if epoch_idx >= len(metrics_df):
                        self.logger.warning(
                            f"[muted]Epoch {epoch} not found in[/] [warning]{run_dir.name}[/]"
                        )
                        continue
                        
                    # Create row with parameters and metric value
                    row_data = {
                        'n_train': int(params['n_train']),
                        'lmax': int(params['lmax']),
                        'metric_value': metrics_df.iloc[epoch_idx][metric]
                    }
                    all_data.append(row_data)
                    
                except Exception as e:
                    self.logger.error(f"Error processing {run_dir.name}: {str(e)}")
                
                progress.advance(task)
        
        finally:
            if not progress:
                progress.stop()
        
        if not all_data:
            self.logger.warning(f"[warning]⚠ No valid data found for plotting {metric}[/]")
            return

        self.logger.info(f"Plotting data from [highlight]{len(all_data)}[/] runs")

        # Create DataFrame and sort by n_train for proper line plotting
        plot_df = pd.DataFrame(all_data)
        plot_df = plot_df.sort_values('n_train')
        
        # After loading all data, print unique lmax values
        self.logger.info(f"Found unique lmax values: {plot_df['lmax'].unique()}")
        self.logger.info(f"Data distribution:\n{plot_df.groupby('lmax').size()}")
        
        # Create plot
        plt.figure(figsize=(12, 8))
        for lmax in sorted(plot_df['lmax'].unique()):
            data = plot_df[plot_df['lmax'] == lmax]
            plt.plot(data['n_train'], data['metric_value'], 
                    marker='o', label=f'lmax={lmax}', linewidth=2, markersize=8)

        plt.xscale('log')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('Number of Training Examples', fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.title(f'Learning Curves: {metric.replace("_", " ").title()}', fontsize=14)
        plt.legend(title='lmax', title_fontsize=12, fontsize=10)
        
        # Save plot
        plot_name = f'learning_curves_{metric}_epoch{epoch}.png'
        save_path = self.plots_dir / plot_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        self.logger.success(f"📊 Plot saved as: [highlight]{save_path}[/]")
        plt.close()

    def visualize_results(self):
        """Generate all visualization plots for the experiment."""
        self.logger.section("Visualization Suite Generation")
        
        metrics = [
            'training_loss', 
            'validation_loss',
            'training_f_mae',
            'validation_f_mae',
            'training_e_mae',
            'validation_e_mae'
        ]
        
        self.logger.info(f"Processing [highlight]{len(metrics)}[/] metrics")
        
        with self.logger.create_progress() as progress:
            task = progress.add_task(
                "[cyan]Generating plots...", 
                total=len(metrics) * 2
            )
            
            for metric in metrics:
                try:
                    # Generate plots for both final epoch and early epochs
                    self.plot_learning_curves(metric=metric, epoch=-1, progress=progress)
                    progress.advance(task)
                    
                    # self.plot_learning_curves(metric=metric, epoch=10, progress=progress)
                    # progress.advance(task)
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate plots for {metric}: {str(e)}")
        
        self.logger.success("\n[success]✨ Visualization suite generation completed[/]")
