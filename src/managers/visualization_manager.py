# src/managers/visualization_manager.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

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

        # Import expected parameter ranges
        from src.settings import PARAM_GRID
        expected_lmax = sorted(PARAM_GRID['lmax'])
        expected_n_train = sorted(PARAM_GRID['n_train'])
        expected_inv_layers = sorted(PARAM_GRID['inv_layers'])

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
        
        # Check for missing configurations
        actual_lmax = sorted(plot_df['lmax'].unique())
        actual_n_train = sorted(plot_df['n_train'].unique())
        actual_inv_layers = sorted(plot_df['inv_layers'].unique())

        # Log any missing configurations
        missing_lmax = set(expected_lmax) - set(actual_lmax)
        missing_n_train = set(expected_n_train) - set(actual_n_train)
        missing_inv_layers = set(expected_inv_layers) - set(actual_inv_layers)

        if missing_lmax:
            self.logger.warning(f"Missing data for L values: {missing_lmax}")
        if missing_n_train:
            self.logger.warning(f"Missing data for n_train values: {missing_n_train}")
        if missing_inv_layers:
            self.logger.warning(f"Missing data for inv_layers values: {missing_inv_layers}")

        # Modern style configuration
        plt.style.use('seaborn-v0_8')  # Modern base style
        sns.set_style("whitegrid", {
            'grid.linestyle': '--',
            'grid.alpha': 0.6,
            'axes.facecolor': 'white',
            'axes.grid': True,
        })
        
        # Custom color palette - modern, professional colors
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f1c40f", "#1abc9c"]
        sns.set_palette(colors)

        # Use expected_lmax for color palette instead of just available values
        color_palette = sns.color_palette("husl", n_colors=len(expected_lmax))
        lmax_color_dict = dict(zip(expected_lmax, color_palette))

        def plot_with_fit(data, x, y, hue, color, **kwargs):
            # Iterate over each unique l value
            for l_val in sorted(data[hue].unique()):
                current_color = color.get(l_val, "black")  # Get the specific color for the current l value
                mask = data[hue] == l_val  # Filter data for current l value
                x_data = np.log10(data[mask][x])
                y_data = data[mask][y]
                
                # Plot scatter points for current l value
                plt.scatter(data[mask][x], y_data, color=current_color, alpha=0.6)
                
                # Only compute a line if there's enough data (more than one point)
                if len(x_data) > 1:
                    # Perform linear regression on log-transformed data for the current l value
                    slope, intercept = np.polyfit(x_data, np.log10(y_data), 1)
                    x_fit = np.logspace(
                        np.log10(min(data[mask][x])), 
                        np.log10(max(data[mask][x])), 
                        100
                    )
                    y_fit = 10**(slope * np.log10(x_fit) + intercept)
                    
                    # Plot the best fit line for current l value
                    plt.plot(
                        x_fit, y_fit, color=current_color, 
                        label=f'L={l_val} (slope={slope:.2f})'
                    )

        # Create FacetGrid
        g = sns.FacetGrid(
            plot_df,
            col="inv_layers",
            height=6,
            aspect=1.2,
            sharey=True,
            sharex=True
        )

        # Apply the plotting function
        g.map_dataframe(
            plot_with_fit,
            x="n_train",
            y="metric_value",
            hue="lmax",
            color=lmax_color_dict
        )

        # Set scales and labels
        for ax in g.axes.flat:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(title="L value (power law fit)", bbox_to_anchor=(1.05, 1))

        # Enhance the plot aesthetics
        g.set_titles(col_template="Invariant Layers: {col_name}", size=12, pad=15)
        g.set_axis_labels("Number of Training Examples", metric.replace("_", " ").title())
        
        # Add a descriptive super title
        plt.suptitle(
            f"{metric.replace('_', ' ').title()} vs Training Size\n",
            y=1.05, 
            fontsize=14, 
            fontweight='bold'
        )

        # Adjust layout
        g.fig.subplots_adjust(top=0.85, wspace=0.3)

        # Enhance each subplot
        for ax in g.axes.flat:
            # Improve tick labels
            ax.tick_params(labelsize=10)
            # Add subtle spines
            for spine in ax.spines.values():
                spine.set_color('#666666')
                spine.set_linewidth(0.8)
            # Improve grid
            ax.grid(True, alpha=0.3, linestyle='--')

        # 5. Save figure
        filename = f"param_comparison_{metric}_epoch{epoch}.png"
        save_path = self.plots_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.success(f"Saved comparison plot to {save_path}")

    def plot_training_time_scaling(self,
                             epoch: int = -1,
                             fixed_params: Optional[Dict[str, str]] = None):
        """
        [RENAMED from plot_wall_time_comparison]
        Plot how training time scales with dataset size across different model configurations.
        """
        if fixed_params is None:
            fixed_params = {}

        self.logger.section(f"Plot Training Time Scaling (epoch={epoch})")

        # 1. Collect run directories (same as before)
        run_dirs = [
            d for d in self.version_dir.iterdir() 
            if d.is_dir() and d.name not in 
               ['configs', 'logs', 'metrics', 'plots', 'checkpoints'] 
               and ("processed_dataset" not in d.name)
        ]

        # 2. Parse data
        all_data = []
        for run_dir in run_dirs:
            metrics_df = self._load_metrics(run_dir)
            if metrics_df.empty or 'wall' not in metrics_df.columns:
                continue

            # Choose epoch row
            if epoch == -1:
                row_idx = -1
            else:
                if epoch >= len(metrics_df):
                    continue
                row_idx = epoch
            
            params = self._parse_run_name(run_dir.name)

            # Check fixed params
            if not all(params.get(k) == v for k, v in fixed_params.items()):
                continue

            # Build row data
            row_data = {
                'wall_time': metrics_df.iloc[row_idx]['wall'],  # Time in seconds
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

        # 3. Create DataFrame
        plot_df = pd.DataFrame(all_data)
        
        # Modern style configuration
        plt.style.use('seaborn-v0_8')
        sns.set_style("whitegrid", {
            'grid.linestyle': '--',
            'grid.alpha': 0.6,
            'axes.facecolor': 'white',
        })

        # Define color palette for L_max values
        lmax_values = sorted(plot_df['lmax'].unique())
        color_palette = sns.color_palette("husl", n_colors=len(lmax_values))
        lmax_color_dict = dict(zip(lmax_values, color_palette))

        # Create plot
        g = sns.relplot(
            data=plot_df,
            x="n_train",
            y="wall_time",
            hue="lmax",
            hue_order=lmax_values,
            palette=lmax_color_dict,
            col="inv_layers",
            kind="line",
            height=6,
            aspect=1.2,
            marker="o",
            markersize=10,
            linewidth=2.5,
            facet_kws={
                "sharey": True,
                "sharex": True,
                "despine": False,
            }
        )

        # Customize plot
        g.set(xscale='log')
        g.set_titles(col_template="Invariant Layers: {col_name}", size=12, pad=15)
        g.set_axis_labels("Number of Training Examples", "Wall Clock Time (seconds)")
        
        g._legend.set_title("L_max", prop={'size': 11, 'weight': 'bold'})
        plt.setp(g._legend.get_texts(), fontsize=10)
        
        plt.suptitle(
            f"Training Time vs Dataset Size\n",
            y=1.05, 
            fontsize=14, 
            fontweight='bold'
        )

        # Enhance subplots
        for ax in g.axes.flat:
            ax.tick_params(labelsize=10)
            for spine in ax.spines.values():
                spine.set_color('#666666')
                spine.set_linewidth(0.8)
            ax.grid(True, alpha=0.3, linestyle='--')

        # Save figure
        filename = f"training_time_scaling_epoch{epoch}.png"
        save_path = self.plots_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.success(f"Saved training time scaling plot to {save_path}")

    def plot_epoch_time_breakdown(self, epoch: int = -1):
        """
        Create stacked bar visualization showing time breakdown per configuration.
        
        Shows:
        - Average time per epoch for each configuration
        - Grouped by training set size (n_train)
        - Stacked by angular momentum (lmax)
        - Separate charts for different numbers of invariant layers
        """
        run_dirs = [
            d for d in self.version_dir.iterdir() 
            if d.is_dir() and d.name not in 
               ['configs', 'logs', 'metrics', 'plots', 'checkpoints'] 
               and ("processed_dataset" not in d.name)
        ]

        all_data = []
        for run_dir in run_dirs:
            metrics_df = self._load_metrics(run_dir)
            if metrics_df.empty or 'wall' not in metrics_df.columns:
                continue

            params = self._parse_run_name(run_dir.name)
            
            # Calculate time per epoch
            if epoch == -1:
                total_time = metrics_df.iloc[-1]['wall']
                num_epochs = len(metrics_df)
            else:
                if epoch >= len(metrics_df):
                    continue
                total_time = metrics_df.iloc[epoch]['wall']
                num_epochs = epoch + 1

            time_per_epoch = total_time / num_epochs
            
            row_data = {
                'time_per_epoch': time_per_epoch,
                **{k: int(v) if k in ['n_train', 'lmax', 'inv_layers'] else v 
                   for k, v in params.items()}
            }
            all_data.append(row_data)

        if not all_data:
            self.logger.warning("No data available for plotting.")
            return

        plot_df = pd.DataFrame(all_data)

        # Create subplot for each inv_layers value with shared y-axis
        inv_layers_values = sorted(plot_df['inv_layers'].unique())
        fig, axes = plt.subplots(1, len(inv_layers_values), 
                                figsize=(6*len(inv_layers_values), 8),
                                sharey=True)  # Add sharey=True here
        if len(inv_layers_values) == 1:
            axes = [axes]

        # Color palette for lmax values
        lmax_values = sorted(plot_df['lmax'].unique())
        colors = sns.color_palette("husl", n_colors=len(lmax_values))

        # Calculate global max y value for consistent scaling
        max_y = 0
        for inv_layers in inv_layers_values:
            data = plot_df[plot_df['inv_layers'] == inv_layers]
            layer_max = 0
            for lmax in lmax_values:
                lmax_data = data[data['lmax'] == lmax].groupby('n_train')['time_per_epoch'].mean()
                layer_max += lmax_data.max() if not lmax_data.empty else 0
            max_y = max(max_y, layer_max)

        for ax, inv_layers in zip(axes, inv_layers_values):
            data = plot_df[plot_df['inv_layers'] == inv_layers]
            
            # Create stacked bars
            bottom = np.zeros(len(data['n_train'].unique()))
            for lmax, color in zip(lmax_values, colors):
                lmax_data = data[data['lmax'] == lmax].groupby('n_train')['time_per_epoch'].mean()
                ax.bar(lmax_data.index.astype(str), lmax_data.values, 
                      bottom=bottom, label=f'L={lmax}', color=color)
                bottom += lmax_data.values

            ax.set_title(f'Invariant Layers: {inv_layers}')
            ax.set_xlabel('Training Set Size')
            if ax == axes[0]:  # Only set ylabel for first subplot
                ax.set_ylabel('Time per Epoch (seconds)')
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim(0, max_y * 1.1)  # Set consistent y limit with 10% padding

        plt.legend(title='Angular Momentum (L)', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        save_path = self.plots_dir / f'epoch_time_breakdown_epoch{epoch}.png'
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.success(f"Saved epoch time breakdown plot to {save_path}")

    def plot_computational_efficiency_heatmap(self, epoch: int = -1):
        """
        Create heatmap visualization showing computational efficiency across configurations.
        
        Shows:
        - Heatmap of time per training sample
        - X-axis: Angular momentum (lmax)
        - Y-axis: Training set size (n_train)
        - Separate plots for different numbers of invariant layers
        - Color intensity: Time per sample (darker = more time)
        """
        run_dirs = [
            d for d in self.version_dir.iterdir() 
            if d.is_dir() and d.name not in 
               ['configs', 'logs', 'metrics', 'plots', 'checkpoints'] 
               and ("processed_dataset" not in d.name)
        ]

        all_data = []
        for run_dir in run_dirs:
            metrics_df = self._load_metrics(run_dir)
            if metrics_df.empty or 'wall' not in metrics_df.columns:
                continue

            params = self._parse_run_name(run_dir.name)
            
            # Calculate time per sample
            if epoch == -1:
                total_time = metrics_df.iloc[-1]['wall']
            else:
                if epoch >= len(metrics_df):
                    continue
                total_time = metrics_df.iloc[epoch]['wall']
            
            n_train = int(params.get('n_train', 0))
            if n_train > 0:
                time_per_sample = total_time / n_train
                row_data = {
                    'time_per_sample': time_per_sample,
                    **{k: int(v) if k in ['n_train', 'lmax', 'inv_layers'] else v 
                       for k, v in params.items()}
                }
                all_data.append(row_data)

        if not all_data:
            self.logger.warning("No data available for plotting.")
            return

        plot_df = pd.DataFrame(all_data)

        # Create heatmap for each inv_layers value
        inv_layers_values = sorted(plot_df['inv_layers'].unique())
        fig, axes = plt.subplots(1, len(inv_layers_values), 
                                figsize=(6*len(inv_layers_values), 8))
        if len(inv_layers_values) == 1:
            axes = [axes]

        for ax, inv_layers in zip(axes, inv_layers_values):
            data = plot_df[plot_df['inv_layers'] == inv_layers]
            
            # Pivot data for heatmap
            heatmap_data = data.pivot(
                index='n_train', 
                columns='lmax', 
                values='time_per_sample'
            )
            
            # Plot heatmap
            sns.heatmap(heatmap_data, ax=ax, 
                        cmap='YlOrRd', 
                        annot=True, 
                        fmt='.2e',
                        cbar_kws={'label': 'Time per Sample (s)'})
            
            ax.set_title(f'Invariant Layers: {inv_layers}')
            ax.set_xlabel('Angular Momentum (L)')
            ax.set_ylabel('Training Set Size')

        plt.tight_layout()
        
        save_path = self.plots_dir / f'computational_efficiency_heatmap_epoch{epoch}.png'
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.success(f"Saved computational efficiency heatmap to {save_path}")

    def plot_accuracy_vs_compute_tradeoff(self, 
                                        metric: str = 'validation_f_mae',
                                        epoch: int = -1):
        """
        Create scatter plot showing accuracy vs computational cost trade-off.
        
        Shows:
        - X-axis: Wall clock time
        - Y-axis: Chosen validation metric
        - Point size: Training set size
        - Point color: Angular momentum (lmax)
        - Point shape: Number of invariant layers
        - Helps identify optimal configurations balancing accuracy and computational cost
        
        Args:
            metric: Validation metric to plot (e.g., 'validation_f_mae', 'validation_loss')
            epoch: Which epoch to analyze (-1 for final epoch)
        """
        run_dirs = [
            d for d in self.version_dir.iterdir() 
            if d.is_dir() and d.name not in 
               ['configs', 'logs', 'metrics', 'plots', 'checkpoints'] 
               and ("processed_dataset" not in d.name)
        ]

        all_data = []
        for run_dir in run_dirs:
            metrics_df = self._load_metrics(run_dir)
            if metrics_df.empty or 'wall' not in metrics_df.columns or metric not in metrics_df.columns:
                continue

            params = self._parse_run_name(run_dir.name)
            
            # Get metric and time values
            if epoch == -1:
                row_idx = -1
            else:
                if epoch >= len(metrics_df):
                    continue
                row_idx = epoch
                
            row_data = {
                'wall_time': metrics_df.iloc[row_idx]['wall'],
                'metric_value': metrics_df.iloc[row_idx][metric],
                **{k: int(v) if k in ['n_train', 'lmax', 'inv_layers'] else v 
                   for k, v in params.items()}
            }
            all_data.append(row_data)

        if not all_data:
            self.logger.warning("No data available for plotting.")
            return

        plot_df = pd.DataFrame(all_data)

        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Color scheme for lmax values
        lmax_values = sorted(plot_df['lmax'].unique())
        colors = sns.color_palette("husl", n_colors=len(lmax_values))
        
        # Markers for inv_layers
        markers = ['o', 's', '^', 'D', 'v']
        inv_layers_values = sorted(plot_df['inv_layers'].unique())
        
        # Plot points
        for inv_layers, marker in zip(inv_layers_values, markers):
            for lmax, color in zip(lmax_values, colors):
                mask = (plot_df['inv_layers'] == inv_layers) & (plot_df['lmax'] == lmax)
                data = plot_df[mask]
                
                plt.scatter(data['wall_time'], 
                           data['metric_value'],
                           s=data['n_train']/10,  # Scale point size
                           c=[color],
                           marker=marker,
                           alpha=0.7,
                           label=f'L={lmax}, Inv={inv_layers}')

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Wall Clock Time (s)')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title('Accuracy vs Computational Cost Trade-off')
        
        # Add legend
        plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add grid
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        save_path = self.plots_dir / f'accuracy_compute_tradeoff_{metric}_epoch{epoch}.png'
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.success(f"Saved accuracy vs compute trade-off plot to {save_path}")

    def plot_slope_vs_inv_layers(self, metric: str = 'validation_loss', epoch: int = -1):
        """
        Create a plot showing how the regression slope of log(metric) versus log(n_train)
        evolves as the network transitions from fully invariant to partially invariant configurations.
        
        For runs with lmax==0 (fully invariant), we set the effective number of equivariant layers to 0 
        (base case). For runs with lmax > 0, the effective number of equivariant layers is computed as:
        
            num_equivariant = TOTAL_LAYERS - inv_layers

        This way, the x-axis (labeled as "Number of Equivariant Layers") is 0 for the fully invariant model,
        and increases as fewer layers become invariant.
        
        The slope is computed via linear regression in the log-log domain for each group of runs (grouped by the
        computed number of equivariant layers) provided that there is sufficient variation in training set size.
        
        Args:
            metric: The metric to analyze (default 'validation_loss').
            epoch: Which epoch to use for metric extraction (-1 for the final epoch).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        from src.settings import TOTAL_LAYERS

        self.logger.section(f"Plotting Slope vs Equivariant Layers for metric='{metric}' (epoch={epoch})")

        # Collect run directories (excluding non-run folders)
        run_dirs = [
            d for d in self.version_dir.iterdir()
            if d.is_dir() and d.name not in ['configs', 'logs', 'metrics', 'plots', 'checkpoints']
               and ("processed_dataset" not in d.name)
        ]
        if not run_dirs:
            self.logger.warning("No run directories found for slope analysis.")
            return

        # Gather data points with effective number of equivariant layers.
        # For lmax==0 (fully invariant), we set num_equivariant = 0.
        # For lmax > 0, compute num_equivariant = TOTAL_LAYERS - inv_layers.
        data_points = []
        for run_dir in run_dirs:
            metrics_df = self._load_metrics(run_dir)
            if metrics_df.empty or metric not in metrics_df.columns:
                self.logger.warning(f"Skipping {run_dir.name}: missing metric data.")
                continue

            if epoch == -1:
                row_idx = -1
            else:
                if epoch >= len(metrics_df):
                    self.logger.warning(f"Epoch {epoch} out of range in {run_dir.name}.")
                    continue
                row_idx = epoch

            params = self._parse_run_name(run_dir.name)
            try:
                n_train = int(params.get('n_train', 0))
                inv_layers = int(params.get('inv_layers', -1))
                lmax = int(params.get('lmax', -1))
            except ValueError:
                self.logger.warning(f"Non-integer parameter encountered in {run_dir.name}")
                continue

            if n_train <= 0 or inv_layers < 0 or lmax < 0:
                self.logger.warning(f"Incomplete parameter set in {run_dir.name}")
                continue

            # For fully invariant configurations (lmax==0), the run is forced to be fully invariant.
            if lmax == 0:
                num_equivariant = 0
            else:
                num_equivariant = TOTAL_LAYERS - inv_layers

            data_points.append({
                'n_train': n_train,
                'num_equivariant': num_equivariant,
                'metric_value': metrics_df.iloc[row_idx][metric]
            })

        if not data_points:
            self.logger.warning("No valid data points were collected for slope analysis.")
            return

        df = pd.DataFrame(data_points)

        # Group by effective number of equivariant layers and compute regression if possible.
        slope_records = []
        for num_eq, group in df.groupby('num_equivariant'):
            if group['n_train'].nunique() < 2:
                self.logger.warning(f"Not enough n_train variations for num_equivariant={num_eq}")
                continue

            x = np.log10(group['n_train'])
            y = np.log10(group['metric_value'])
            slope, intercept = np.polyfit(x, y, 1)
            slope_records.append({
                'num_equivariant': num_eq,
                'slope': slope
            })

        if not slope_records:
            self.logger.warning("No slopes computed; insufficient data across groups.")
            return

        summary = pd.DataFrame(slope_records).sort_values('num_equivariant')

        plt.figure(figsize=(8, 6))
        plt.plot(summary['num_equivariant'], summary['slope'], '-o', markersize=8, label='Slope')
        plt.xlabel("Number of Equivariant Layers (0 = Fully Invariant)")
        plt.ylabel("Slope (log(metric) vs. log(n_train))")
        plt.title(f"Evolution of Slope vs. Equivariant Layers\nMetric: {metric}, Epoch: {epoch if epoch != -1 else 'Final'}")
        plt.grid(True, alpha=0.3)
        plt.legend()

        filename = f"slope_vs_equivariant_layers_{metric}_epoch{epoch}.png"
        save_path = self.plots_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.success(f"Saved slope vs. equivariant layers plot to {save_path}")

    def visualize_results(self):
        """
        Generate a comprehensive visualization suite with enhanced aesthetics.
        """
        self.logger.section("Visualization Suite Generation")

        # Define metrics with more readable display names
        metrics_config = {
            'training_loss': 'Training Loss',
            'validation_loss': 'Validation Loss',
            'training_f_mae': 'Training Force MAE',
            'validation_f_mae': 'Validation Force MAE',
            'training_e_mae': 'Training Energy MAE',
            'validation_e_mae': 'Validation Energy MAE'
        }

        # Set global style configurations
        plt.style.use('seaborn-v0_8')
        sns.set_style("whitegrid", {
            'grid.linestyle': '--',
            'grid.alpha': 0.6,
            'axes.facecolor': 'white',
        })
        
        # Set modern color palette
        colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c", "#f1c40f", "#1abc9c"]
        sns.set_palette(colors)

        self.logger.info(f"Generating enhanced plots for {len(metrics_config)} metrics...")

        with self.logger.create_progress() as progress:
            task = progress.add_task(
                "[cyan]Generating publication-quality plots...", 
                total=len(metrics_config)
            )
            
            for metric, display_name in metrics_config.items():
                try:
                    # self.plot_param_comparison(
                    #     metric=metric,
                    #     epoch=-1,
                    #     fixed_params={}
                    # )
                    self.plot_slope_vs_inv_layers(
                        metric=metric,
                        epoch=-1
                    )
                except Exception as e:
                    self.logger.error(f"Failed to generate plots for {display_name}: {str(e)}")
                
                progress.advance(task)

        self.logger.success("Enhanced visualization suite generation completed! 🎨")
