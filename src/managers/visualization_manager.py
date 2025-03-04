# src/managers/visualization_manager.py
"""
VisualizationManager module:
This module generates plots from the centralized experiment DataFrame.
It supports filtering by hyperparameters (including dataset) and creates publication-quality plots.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta
import matplotlib.dates as mdates
from src.managers.logging_manager import LoggingManager
from src.managers.experiment_tracker import ExperimentTracker
import re

class VisualizationManager:
    def __init__(self, df: pd.DataFrame = None):
        """
        Initialize with a centralized experiment DataFrame.
        If no DataFrame is provided, load it via ExperimentTracker.
        """
        self.logger = LoggingManager()
        if df is None:
            tracker = ExperimentTracker()
            self.df = tracker.load_experiment_df()
        else:
            self.df = df
        
        # Convert timestamp strings to datetime objects if they aren't already
        if 'timestamp' in self.df.columns and isinstance(self.df['timestamp'].iloc[0], str):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        sns.set_theme(style="whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.figsize'] = [10, 6]

    def filter_experiments(self, **filters):
        """
        Filter the experiment DataFrame based on given column filters.
        Usage example: filter_experiments(dataset="ds1", lmax=2)
        """
        df_filtered = self.df.copy()
        for key, value in filters.items():
            df_filtered = df_filtered[df_filtered[key] == value]
        return df_filtered
    
    def filter_by_date(self, start_date=None, end_date=None):
        """
        Filter experiments by date range.
        Dates can be provided as strings ('YYYY-MM-DD') or datetime objects.
        """
        df_filtered = self.df.copy()
        
        if start_date:
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            df_filtered = df_filtered[df_filtered['timestamp'] >= start_date]
            
        if end_date:
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            df_filtered = df_filtered[df_filtered['timestamp'] <= end_date]
            
        return df_filtered
    
    def get_available_datasets(self):
        """Return a list of all available datasets in the experiment DataFrame."""
        return self.df['dataset'].unique().tolist()
    
    def get_available_metrics(self):
        """Return a list of all available metrics columns in the experiment DataFrame."""
        # Define what columns are considered metrics
        metric_patterns = ['_mae', '_loss', 'wall_time', 'time_per_epoch', 'regression_slope']
        return [col for col in self.df.columns if any(pattern in col for pattern in metric_patterns)]
    
    def get_date_range(self):
        """Return the earliest and latest dates in the experiment DataFrame."""
        if 'timestamp' not in self.df.columns:
            return None, None
        return self.df['timestamp'].min(), self.df['timestamp'].max()
        
    def plot_param_comparison(self, metric: str, fixed_params: dict = {}):
        """
        Generate a plot comparing hyperparameters against a selected metric.
        Filters experiments based on fixed_params.
        """
        self.logger.section(f"Plot Param Comparison for metric='{metric}'")
        plot_df = self.df.copy()
        for key, value in fixed_params.items():
            plot_df = plot_df[plot_df[key] == value]
        
        plot_df = plot_df[plot_df[metric].notnull()]
        if plot_df.empty:
            self.logger.warning("No data left after filtering; nothing to plot.")
            return

        # Create a FacetGrid faceted by inv_layers.
        g = sns.FacetGrid(
            plot_df,
            col="inv_layers",
            height=6,
            aspect=1.2,
            sharey=True,
            sharex=True
        )

        def scatter_with_fit(data, x, y, hue, color, **kwargs):
            for l_val in sorted(data[hue].unique()):
                current_color = color.get(l_val, "black")
                mask = data[hue] == l_val
                x_data = np.log10(data[mask][x])
                y_data = np.log10(data[mask][y])
                plt.scatter(data[mask][x], data[mask][y], color=current_color, alpha=0.6)
                if len(x_data) > 1:
                    slope, intercept = np.polyfit(x_data, y_data, 1)
                    x_fit = np.logspace(np.log10(min(data[mask][x])), np.log10(max(data[mask][x])), 100)
                    y_fit = 10**(slope * np.log10(x_fit) + intercept)
                    plt.plot(x_fit, y_fit, color=current_color, label=f'L={l_val} (slope={slope:.2f})')

        expected_lmax = sorted(plot_df["lmax"].dropna().unique())
        palette = sns.color_palette("husl", n_colors=len(expected_lmax))
        lmax_color_dict = dict(zip(expected_lmax, palette))

        g.map_dataframe(scatter_with_fit, x="n_train", y=metric, hue="lmax", color=lmax_color_dict)

        for ax in g.axes.flat:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.legend(title="L value (fit)", bbox_to_anchor=(1.05, 1))
        g.set_titles(col_template="Invariant Layers: {col_name}")
        g.set_axis_labels("Number of Training Examples", metric.replace("_", " ").title())
        plt.suptitle(f"{metric.replace('_', ' ').title()} vs Training Size", y=1.05, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plots_dir = Path("src/results") / self.df["experiment_name"].iloc[0] / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        save_path = plots_dir / f"param_comparison_{metric}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.success(f"Saved param comparison plot to {save_path}")

    def plot_by_dataset(self, metric: str, x_axis: str = "n_train", hue: str = "lmax", 
                        facet_by: str = "inv_layers", date_range: tuple = None):
        """
        Generate a plot comparing a metric across different datasets or parameters.
        
        Args:
            metric (str): The metric to plot (e.g., 'final_validation_f_mae')
            x_axis (str): The parameter to use for the x-axis (default: 'n_train')
            hue (str): The parameter to use for color coding (default: 'lmax')
            facet_by (str): Parameter to create facet grid (default: 'inv_layers')
            date_range (tuple): Optional (start_date, end_date) for filtering
        """
        self.logger.section(f"Plotting {metric} by {x_axis}, colored by {hue}")
        
        # Create a filtered copy of the dataframe
        plot_df = self.df.copy()
        
        # Filter by date if specified
        if date_range:
            start_date, end_date = date_range
            plot_df = plot_df[(plot_df['timestamp'] >= start_date) & 
                              (plot_df['timestamp'] <= end_date)]
                              
        # Add friendly display name if experiment_id is being used
        if x_axis == 'experiment_id' or hue == 'experiment_id' or facet_by == 'experiment_id':
            plot_df['display_name'] = plot_df['experiment_id'].apply(self._get_friendly_experiment_id)
            
            # Replace experiment_id with display_name in the parameters
            if x_axis == 'experiment_id':
                x_axis = 'display_name'
            if hue == 'experiment_id':
                hue = 'display_name'
            if facet_by == 'experiment_id':
                facet_by = 'display_name'
        
        # Skip if we don't have the metric or axis columns
        if metric not in plot_df.columns or x_axis not in plot_df.columns:
            self.logger.error(f"Missing columns: metric={metric}, x_axis={x_axis}")
            return
        
        # Skip if we don't have at least 2 data points
        valid_data = plot_df.dropna(subset=[metric, x_axis])
        if len(valid_data) < 2:
            self.logger.warning(f"Not enough data points for plotting {metric} vs {x_axis}")
            return
        
        # Remove rows with null values in key columns
        plot_df = plot_df[plot_df[metric].notnull() & 
                          plot_df[x_axis].notnull() & 
                          plot_df[hue].notnull() & 
                          plot_df[facet_by].notnull() &
                          plot_df['dataset'].notnull()]
        
        if plot_df.empty:
            self.logger.warning("No data left after filtering; nothing to plot.")
            return
            
        # Get unique datasets
        datasets = plot_df['dataset'].unique()
        self.logger.info(f"Creating plots for datasets: {', '.join(datasets)}")
        
        # Plot settings
        fig, axes = plt.subplots(1, len(datasets), figsize=(7*len(datasets), 6), sharey=True)
        if len(datasets) == 1:
            axes = [axes]  # Make sure axes is always iterable
            
        # Create subplots for each dataset
        for i, dataset in enumerate(datasets):
            dataset_df = plot_df[plot_df['dataset'] == dataset]
            
            # Group by facet_by parameter
            for facet_val in sorted(dataset_df[facet_by].unique()):
                facet_df = dataset_df[dataset_df[facet_by] == facet_val]
                
                # Create scatter plot
                for hue_val in sorted(facet_df[hue].unique()):
                    hue_df = facet_df[facet_df[hue] == hue_val]
                    axes[i].scatter(hue_df[x_axis], hue_df[metric], 
                                   label=f"{facet_by}={facet_val}, {hue}={hue_val}",
                                   alpha=0.7)
                    
                    # Add best fit line if enough data points
                    if len(hue_df) > 1:
                        try:
                            # Use log scale for fitting if x_axis is typically log-scaled
                            if x_axis in ['n_train', 'num_features']:
                                x_data = np.log10(hue_df[x_axis])
                                y_data = np.log10(hue_df[metric])
                                slope, intercept = np.polyfit(x_data, y_data, 1)
                                
                                x_fit = np.logspace(np.log10(min(hue_df[x_axis])), 
                                                   np.log10(max(hue_df[x_axis])), 100)
                                y_fit = 10**(slope * np.log10(x_fit) + intercept)
                            else:
                                x_data = hue_df[x_axis]
                                y_data = hue_df[metric]
                                slope, intercept = np.polyfit(x_data, y_data, 1)
                                
                                x_fit = np.linspace(min(hue_df[x_axis]), max(hue_df[x_axis]), 100)
                                y_fit = slope * x_fit + intercept
                                
                            axes[i].plot(x_fit, y_fit, '--', alpha=0.5)
                        except Exception as e:
                            self.logger.warning(f"Could not fit line for {dataset}, {facet_by}={facet_val}, {hue}={hue_val}: {e}")
            
            # Set axis properties
            if x_axis in ['n_train', 'num_features']:
                axes[i].set_xscale('log')
            if metric.endswith('_mae') or metric.endswith('_loss'):
                axes[i].set_yscale('log')
                
            axes[i].set_title(f"Dataset: {dataset}")
            axes[i].set_xlabel(x_axis.replace('_', ' ').title())
            axes[i].grid(True, alpha=0.3, linestyle='--')
            
        # Set common y-label
        fig.text(0.04, 0.5, metric.replace('_', ' ').title(), va='center', rotation='vertical', fontsize=12)
        
        # Add legend
        plt.figlegend(loc='upper center', bbox_to_anchor=(0.5, 0), ncol=min(5, len(plot_df[facet_by].unique()) * len(plot_df[hue].unique())))
        
        plt.suptitle(f"{metric.replace('_', ' ').title()} Comparison Across Datasets", 
                    fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        # Save plot
        plots_dir = Path("src/results") / self.df["experiment_name"].iloc[0] / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        save_path = plots_dir / f"dataset_comparison_{metric}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.success(f"Saved dataset comparison plot to {save_path}")
        
    def plot_metric_over_time(self, metric: str, group_by: str = 'dataset', 
                             start_date: str = None, end_date: str = None,
                             rolling_window: int = None):
        """
        Plot how a metric has changed over time.
        
        Args:
            metric: The metric to track over time
            group_by: The parameter to group experiments by (e.g., 'dataset', 'lmax')
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            rolling_window: If provided, apply a rolling average with this window size
        """
        self.logger.section(f"Time Series Plot for metric='{metric}'")
        
        # Filter by date if specified
        plot_df = self.filter_by_date(start_date, end_date)
        
        # Check if we have timestamp column
        if 'timestamp' not in plot_df.columns:
            self.logger.error("Cannot create time series plot: no timestamp column found")
            return
            
        # Remove rows with null values in key columns
        plot_df = plot_df[plot_df[metric].notnull() & 
                          plot_df[group_by].notnull() & 
                          plot_df['timestamp'].notnull()]
        
        if plot_df.empty:
            self.logger.warning("No data left after filtering; nothing to plot.")
            return
            
        # Sort by timestamp
        plot_df = plot_df.sort_values('timestamp')
        
        # Create the plot
        plt.figure(figsize=(12, 7))
        
        # Group by the specified parameter
        for group_val in sorted(plot_df[group_by].unique()):
            group_df = plot_df[plot_df[group_by] == group_val]
            
            # Apply rolling average if specified
            if rolling_window and len(group_df) > rolling_window:
                group_df[f'{metric}_rolling'] = group_df[metric].rolling(rolling_window).mean()
                plt.plot(group_df['timestamp'], group_df[f'{metric}_rolling'], 
                         label=f"{group_by}={group_val} ({rolling_window}-exp rolling avg)",
                         linewidth=2)
                # Also plot the raw data with lower alpha
                plt.scatter(group_df['timestamp'], group_df[metric], 
                           alpha=0.3, s=30)
            else:
                # Just plot the raw data with lines connecting points
                plt.plot(group_df['timestamp'], group_df[metric],
                        label=f"{group_by}={group_val}",
                        marker='o', linestyle='-', alpha=0.7)
        
        # Format the plot
        plt.xlabel("Date")
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f"{metric.replace('_', ' ').title()} Over Time", fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend()
        
        # Format x-axis date labels
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()  # Rotate date labels
        
        # Add y-axis log scale if appropriate
        if metric.endswith('_mae') or metric.endswith('_loss'):
            plt.yscale('log')
        
        # Save the plot
        plots_dir = Path("src/results") / self.df["experiment_name"].iloc[0] / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        date_suffix = ""
        if start_date or end_date:
            date_suffix = f"_{start_date or 'start'}_{end_date or 'end'}"
        save_path = plots_dir / f"time_series_{metric}{date_suffix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        self.logger.success(f"Saved time series plot to {save_path}")
    
    def plot_parameter_heatmap(self, metric: str, x_param: str, y_param: str, 
                              filter_params: dict = {}):
        """
        Create a heatmap showing how a metric varies with two parameters.
        
        Args:
            metric: The metric to visualize
            x_param: The parameter to use for the x-axis
            y_param: The parameter to use for the y-axis
            filter_params: Dictionary of parameter filters to apply
        """
        self.logger.section(f"Parameter Heatmap for metric='{metric}'")
        
        # Apply filters
        plot_df = self.df.copy()
        for key, value in filter_params.items():
            plot_df = plot_df[plot_df[key] == value]
            
        # Remove rows with null values
        plot_df = plot_df[plot_df[metric].notnull() & 
                          plot_df[x_param].notnull() & 
                          plot_df[y_param].notnull()]
        
        if plot_df.empty:
            self.logger.warning("No data left after filtering; nothing to plot.")
            return
            
        # Create a pivot table for the heatmap
        try:
            pivot_df = plot_df.pivot_table(index=y_param, columns=x_param, values=metric, aggfunc='mean')
            
            # Create the plot
            plt.figure(figsize=(10, 8))
            sns.heatmap(pivot_df, annot=True, cmap="viridis", fmt=".3g", 
                       cbar_kws={'label': metric.replace('_', ' ').title()})
            
            # Format the plot
            plt.title(f"{metric.replace('_', ' ').title()} Heatmap: {y_param.title()} vs {x_param.title()}", 
                     fontsize=14, fontweight='bold')
            plt.xlabel(x_param.replace('_', ' ').title())
            plt.ylabel(y_param.replace('_', ' ').title())
            
            # Add filter info to the plot
            if filter_params:
                filter_text = ", ".join([f"{k}={v}" for k, v in filter_params.items()])
                plt.figtext(0.5, -0.05, f"Filters: {filter_text}", ha='center', fontsize=10)
            
            # Save the plot
            plots_dir = Path("src/results") / self.df["experiment_name"].iloc[0] / "plots"
            plots_dir.mkdir(exist_ok=True, parents=True)
            filter_suffix = "_".join([f"{k}_{v}" for k, v in filter_params.items()]) if filter_params else ""
            if filter_suffix:
                filter_suffix = f"_{filter_suffix}"
            save_path = plots_dir / f"heatmap_{metric}_{x_param}_{y_param}{filter_suffix}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            self.logger.success(f"Saved heatmap plot to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Could not create heatmap: {e}")

    def _get_friendly_experiment_id(self, experiment_id):
        """
        Convert hash-like experiment IDs to more readable names.
        For actual experiment runs with parameters in the name, keep as is.
        For hash-like names, simplify to something more readable.
        """
        # If the experiment_id contains parameters, keep it as is
        if "n_train" in experiment_id or "lmax" in experiment_id:
            return experiment_id
            
        # If it starts with "processed_dataset_", clean it up
        if experiment_id.startswith("processed_dataset_"):
            # Extract hash and shorten it
            hash_part = experiment_id.split("processed_dataset_")[1]
            short_hash = hash_part[:6] if len(hash_part) > 6 else hash_part
            return f"processed_dataset_{short_hash}..."
            
        # If it's a pure hash
        if len(experiment_id) >= 20 and re.match(r'^[a-f0-9]+$', experiment_id):
            return f"dataset_{experiment_id[:6]}..."
            
        # Default: return as is
        return experiment_id

    def get_latest_plot(self):
        """
        Find the most recently created/modified plot file in the results directory.
        
        Returns:
            str: Path to the latest plot file, or None if no plots are found
        """
        try:
            # Determine the plot directory based on the experiment name
            if len(self.df) == 0 or "experiment_name" not in self.df.columns:
                self.logger.error("No experiment data available to locate plots directory")
                return None
                
            experiment_name = self.df["experiment_name"].iloc[0]
            plots_dir = Path("src/results") / experiment_name / "plots"
            
            if not plots_dir.exists():
                self.logger.warning(f"Plots directory not found: {plots_dir}")
                return None
                
            # Find all PNG files in the plots directory
            plot_files = list(plots_dir.glob("*.png"))
            
            if not plot_files:
                self.logger.warning("No plot files found in the plots directory")
                return None
                
            # Get the most recently modified file
            latest_plot = max(plot_files, key=lambda p: p.stat().st_mtime)
            self.logger.info(f"Found latest plot: {latest_plot}")
            
            return str(latest_plot)
            
        except Exception as e:
            self.logger.error(f"Error finding latest plot: {e}")
            return None
