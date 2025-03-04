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
        
        Special handling for is_test flag:
        - If is_test=True, only show test experiments
        - If is_test=False, only show non-test experiments
        - If is_test=None, show all experiments
        """
        df_filtered = self.df.copy()
        for key, value in filters.items():
            # Special handling for is_test to allow explicit None value
            if key == "is_test" and value is None:
                continue  # Skip filtering if None is explicitly passed
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
        Each dataset will be shown in a separate panel, but part of the same plot.
        This ensures that all non-plotting parameters (except date) are kept consistent
        for proper comparison across datasets.
        
        Args:
            metric (str): The metric to plot (e.g., 'final_validation_f_mae')
            x_axis (str): The parameter to use for the x-axis (default: 'n_train')
            hue (str): The parameter to use for color coding (default: 'lmax')
            facet_by (str): Parameter to create subplots within each dataset panel (default: 'inv_layers')
            date_range (tuple): Optional (start_date, end_date) for filtering
        """
        self.logger.section(f"Plotting {metric} by dataset, x-axis: {x_axis}, hue: {hue}, facet: {facet_by}")
        
        # Create a filtered copy of the dataframe
        plot_df = self.df.copy()
        
        # Filter by date if specified
        if date_range:
            start_date, end_date = date_range
            if start_date:
                plot_df = plot_df[plot_df['timestamp'] >= pd.to_datetime(start_date)]
            if end_date:
                plot_df = plot_df[plot_df['timestamp'] <= pd.to_datetime(end_date)]
                              
        # Add friendly display name if experiment_id is being used
        if x_axis == 'experiment_id' or hue == 'experiment_id' or facet_by == 'experiment_id':
            plot_df['display_name'] = plot_df['experiment_id'].apply(
                lambda x: x[:8] + '...' if isinstance(x, str) and len(x) > 10 else x
            )
            
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
                          plot_df[facet_by].notnull()]
        
        if plot_df.empty:
            self.logger.warning("No data left after filtering; nothing to plot.")
            return
            
        # Get unique datasets
        datasets = sorted(plot_df['dataset'].unique())
        self.logger.info(f"Creating plots for datasets: {', '.join(datasets)}")
        
        # Identify all the parameters that should be consistent
        all_columns = set(plot_df.columns)
        plotting_params = {'dataset', x_axis, hue, facet_by, 'timestamp', metric, 'experiment_id', 'display_name'}
        consistent_params = all_columns - plotting_params
        
        # For each consistent parameter, check if values vary and warn if they do
        for param in consistent_params:
            if param in plot_df.columns and not all(pd.isna(plot_df[param])):
                unique_values = plot_df[param].dropna().unique()
                if len(unique_values) > 1:
                    self.logger.warning(f"Parameter '{param}' has multiple values: {unique_values}")
                    self.logger.warning(f"This may affect the validity of dataset comparisons")
        
        # Determine the number of facet values
        facet_values = sorted(plot_df[facet_by].unique())
        
        # Create the figure
        fig_height = 5 * len(facet_values)
        fig_width = 7 * len(datasets)
        fig = plt.figure(figsize=(fig_width, fig_height))
        
        # Create a grid of subplots
        gs = plt.GridSpec(len(facet_values), len(datasets))
        
        # Get color palette for hue values
        hue_values = sorted(plot_df[hue].unique())
        palette = sns.color_palette("husl", n_colors=len(hue_values))
        hue_colors = dict(zip(hue_values, palette))
        
        # Create plots for each dataset and facet value
        for i, facet_val in enumerate(facet_values):
            for j, dataset in enumerate(datasets):
                # Create subplot
                ax = fig.add_subplot(gs[i, j])
                
                # Filter data for this dataset and facet value
                subset = plot_df[(plot_df['dataset'] == dataset) & (plot_df[facet_by] == facet_val)]
                
                if subset.empty:
                    ax.text(0.5, 0.5, f"No data for\n{dataset}\n{facet_by}={facet_val}",
                            ha='center', va='center', fontsize=12)
                    continue
                
                # Plot each hue value
                for hue_val in hue_values:
                    hue_data = subset[subset[hue] == hue_val]
                    if not hue_data.empty:
                        ax.scatter(hue_data[x_axis], hue_data[metric], 
                                  color=hue_colors[hue_val], label=f"{hue}={hue_val}",
                                  alpha=0.7, s=80)
                        
                        # Add best fit line if enough data points
                        if len(hue_data) > 1:
                            try:
                                # Use log scale for fitting if x_axis is typically log-scaled
                                if x_axis in ['n_train', 'num_features']:
                                    x_data = np.log10(hue_data[x_axis])
                                    y_data = np.log10(hue_data[metric])
                                    slope, intercept = np.polyfit(x_data, y_data, 1)
                                    
                                    x_fit = np.logspace(np.log10(min(hue_data[x_axis])), 
                                                      np.log10(max(hue_data[x_axis])), 100)
                                    y_fit = 10**(slope * np.log10(x_fit) + intercept)
                                    
                                    # Display slope in legend
                                    ax.plot(x_fit, y_fit, '--', color=hue_colors[hue_val], alpha=0.7,
                                          label=f"{hue}={hue_val} (slope={slope:.2f})")
                                else:
                                    x_data = hue_data[x_axis]
                                    y_data = hue_data[metric]
                                    slope, intercept = np.polyfit(x_data, y_data, 1)
                                    
                                    x_fit = np.linspace(min(hue_data[x_axis]), max(hue_data[x_axis]), 100)
                                    y_fit = slope * x_fit + intercept
                                    
                                    ax.plot(x_fit, y_fit, '--', color=hue_colors[hue_val], alpha=0.7,
                                           label=f"{hue}={hue_val} (slope={slope:.2f})")
                            except Exception as e:
                                self.logger.warning(f"Could not fit line for {dataset}, {facet_by}={facet_val}, {hue}={hue_val}: {e}")
                
                # Set axis properties
                if x_axis in ['n_train', 'num_features']:
                    ax.set_xscale('log')
                if metric.endswith('_mae') or metric.endswith('_loss'):
                    ax.set_yscale('log')
                
                # Set titles only for certain subplots
                if i == 0:  # First row gets dataset titles
                    ax.set_title(f"Dataset: {dataset}", fontsize=14, fontweight='bold')
                if j == 0:  # First column gets facet labels
                    ax.set_ylabel(f"{facet_by}={facet_val}\n{metric.replace('_', ' ').title()}", fontsize=12)
                
                # Only add x-label to bottom row
                if i == len(facet_values) - 1:
                    ax.set_xlabel(x_axis.replace('_', ' ').title(), fontsize=12)
                
                # Add grid
                ax.grid(True, alpha=0.3, linestyle='--')
                
                # Add legend only to the rightmost plots
                if j == len(datasets) - 1:
                    ax.legend(title=hue.replace('_', ' ').title(), 
                             bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.suptitle(f"{metric.replace('_', ' ').title()} Comparison Across Datasets", 
                    fontsize=16, fontweight='bold', y=1.01)
        plt.tight_layout()
        
        # Save plot
        plots_dir = Path("src/results") / self.df["experiment_name"].iloc[0] / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        save_path = plots_dir / f"dataset_comparison_{metric}_{x_axis}_{hue}_{facet_by}.png"
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

    def group_by_config(self, grouping_params=None, exclude_params=None):
        """
        Group experiments by their configuration parameters.
        
        Args:
            grouping_params (list): Parameters to group by. If None, groups by all model parameters
                                    except experiment_id, timestamp, and metric results.
            exclude_params (list): Parameters to exclude from grouping.
            
        Returns:
            dict: Dictionary mapping parameter combinations to experiment groups
        """
        df = self.df.copy()
        
        # Default parameters to group by (all parameters except metrics and metadata)
        if grouping_params is None:
            # Identify potential grouping parameters (all columns that aren't metrics or metadata)
            all_columns = set(df.columns)
            non_param_cols = {
                'experiment_id', 'timestamp', 'run_directory', 'config_file', 'metrics_file',
                'training_log_file', 'evaluation_log_file', 'deployed_model_file', 'plot_directory',
                'status', 'experiment_notes', 'run_iteration'  # Add run_iteration to columns to exclude
            }
            
            # Also exclude metric columns (typically have 'loss' or 'mae' in the name)
            metric_cols = {col for col in all_columns if 
                         any(metric in col for metric in ['loss', 'mae', 'time', 'epoch'])}
            
            # Parameters to group by are all columns except non-parameters and metrics
            grouping_params = list(all_columns - non_param_cols - metric_cols)
        
        # Further exclude any specified parameters
        if exclude_params:
            grouping_params = [p for p in grouping_params if p not in exclude_params]
            
        self.logger.info(f"Grouping experiments by parameters: {grouping_params}")
        
        # Group experiments by the parameter combination
        experiment_groups = {}
        
        # Handle case where some parameters might be missing in some rows
        df_filled = df.copy()
        for param in grouping_params:
            if param not in df_filled.columns:
                df_filled[param] = np.nan
                
        # Iterate through rows and build groups
        for _, row in df_filled.iterrows():
            # Create a tuple of parameter values to use as dictionary key
            param_values = tuple((param, row[param]) for param in grouping_params)
            
            # Skip rows with missing parameter values
            if any(pd.isna(v) for _, v in param_values):
                continue
                
            # Add row to the appropriate group
            if param_values not in experiment_groups:
                experiment_groups[param_values] = []
            experiment_groups[param_values].append(row)
            
        # Convert lists of rows to DataFrame groups
        for params, rows in experiment_groups.items():
            experiment_groups[params] = pd.DataFrame(rows)
            
        # Log group counts
        self.logger.info(f"Found {len(experiment_groups)} unique parameter combinations")
        for params, group_df in experiment_groups.items():
            param_str = ", ".join(f"{p}={v}" for p, v in params)
            self.logger.info(f"  {param_str}: {len(group_df)} experiments")
            
        return experiment_groups
    
    def calculate_group_statistics(self, experiment_groups, metrics):
        """
        Calculate statistics (mean, std, min, max) for specified metrics across experiment groups.
        
        Args:
            experiment_groups (dict): Groups of experiments with the same parameters
            metrics (list): List of metrics to calculate statistics for
            
        Returns:
            pd.DataFrame: DataFrame with one row per parameter combination, with statistics for each metric
        """
        if not experiment_groups:
            self.logger.warning("No experiment groups provided for statistical analysis")
            return pd.DataFrame()
            
        # Prepare results dataframe
        result_rows = []
        
        for params, group_df in experiment_groups.items():
            # Convert params tuple to dict for easier handling
            param_dict = {p: v for p, v in params}
            
            # Calculate statistics for each metric
            for metric in metrics:
                if metric not in group_df.columns:
                    continue
                    
                # Skip if all values are NaN
                if group_df[metric].isna().all():
                    continue
                
                # Basic statistics
                mean_val = group_df[metric].mean()
                std_val = group_df[metric].std()
                min_val = group_df[metric].min()
                max_val = group_df[metric].max()
                count = group_df[metric].count()
                
                # Only add if we have valid statistics
                if not pd.isna(mean_val) and count > 0:
                    # Create row with parameters and statistics
                    result_row = param_dict.copy()
                    result_row.update({
                        'metric': metric,
                        'mean': mean_val,
                        'std': std_val,
                        'min': min_val,
                        'max': max_val,
                        'count': count,
                        'cv': std_val / mean_val if mean_val != 0 else np.nan,  # Coefficient of variation
                    })
                    result_rows.append(result_row)
        
        # Convert to DataFrame
        stats_df = pd.DataFrame(result_rows)
        return stats_df
        
    def plot_with_error_bars(self, metric, x_axis, hue=None, facet_by=None, filters=None, 
                            group_by=None, exclude_from_grouping=None):
        """
        Generate a plot with error bars showing mean ± std across multiple runs.
        
        Args:
            metric (str): The metric to visualize
            x_axis (str): Parameter to use for x-axis
            hue (str): Parameter to use for color coding
            facet_by (str): Parameter to use for faceting
            filters (dict): Filters to apply before grouping
            group_by (list): Parameters to group by. If None, groups by all model parameters
            exclude_from_grouping (list): Parameters to exclude from grouping
            
        This creates a plot showing mean values with error bars for standard deviation.
        """
        self.logger.section(f"Generating plot with error bars for {metric}")
        
        # Apply filters if provided
        df_filtered = self.df.copy()
        if filters:
            for key, value in filters.items():
                df_filtered = df_filtered[df_filtered[key] == value]
                
        if df_filtered.empty:
            self.logger.warning("No data left after filtering")
            return
            
        # Create a temporary VisualizationManager with filtered data
        temp_viz = VisualizationManager(df_filtered)
        
        # Set default grouping parameters if not provided
        if group_by is None:
            # Group by everything except the parameters used for plotting
            exclude = [x_axis]
            if hue:
                exclude.append(hue)
            if facet_by:
                exclude.append(facet_by)
                
            # Add any additional exclusions
            if exclude_from_grouping:
                exclude.extend(exclude_from_grouping)
                
            # Use all parameters except those for plotting
            group_by = [col for col in df_filtered.columns if col not in exclude]
            
        # Group experiments by configuration
        experiment_groups = temp_viz.group_by_config(group_by, exclude_params=exclude_from_grouping)
        
        # Calculate statistics for the metric
        stats_df = temp_viz.calculate_group_statistics(experiment_groups, [metric])
        
        if stats_df.empty:
            self.logger.warning(f"No statistics available for {metric}")
            return
            
        # Prepare for plotting
        plt.figure(figsize=(12, 8))
        
        # Determine unique values for faceting
        facet_values = [None]
        if facet_by and facet_by in stats_df.columns:
            facet_values = sorted(stats_df[facet_by].unique())
            
        # Create subplots for faceting
        fig, axes = plt.subplots(1, len(facet_values), 
                                figsize=(7*len(facet_values), 6), 
                                sharey=True, squeeze=False)
        axes = axes.flatten()  # Ensure we can index axes as a 1D array
        
        # For each facet value, create a plot with error bars
        for i, facet_val in enumerate(facet_values):
            ax = axes[i]
            
            # Filter data for this facet
            if facet_by and facet_val is not None:
                facet_data = stats_df[stats_df[facet_by] == facet_val]
            else:
                facet_data = stats_df
                
            # Skip if no data for this facet
            if facet_data.empty:
                ax.text(0.5, 0.5, f"No data for\n{facet_by}={facet_val}",
                       ha='center', va='center', fontsize=12)
                continue
                
            # Determine unique values for hue
            hue_values = [None]
            if hue and hue in facet_data.columns:
                hue_values = sorted(facet_data[hue].unique())
                
            # Get color palette for hue
            palette = sns.color_palette("husl", n_colors=len(hue_values))
            colors = dict(zip(hue_values, palette))
            
            # For each hue value, plot points with error bars
            for j, hue_val in enumerate(hue_values):
                # Filter data for this hue
                if hue and hue_val is not None:
                    hue_data = facet_data[facet_data[hue] == hue_val]
                else:
                    hue_data = facet_data
                    
                # Skip if no data for this hue
                if hue_data.empty:
                    continue
                    
                # Sort by x-axis for line plotting
                hue_data = hue_data.sort_values(by=x_axis)
                
                # Get color
                color = colors.get(hue_val, 'blue')
                
                # Plot mean values with error bars
                ax.errorbar(
                    hue_data[x_axis], 
                    hue_data['mean'], 
                    yerr=hue_data['std'],
                    fmt='o-',
                    color=color,
                    capsize=5,
                    label=f"{hue}={hue_val}" if hue else None,
                    alpha=0.8
                )
                
                # Add individual data points if there are fewer than 20
                if len(experiment_groups) < 20:
                    for params, group_df in experiment_groups.items():
                        # Convert params tuple to dict
                        param_dict = {p: v for p, v in params}
                        
                        # Check if this group matches our hue and facet
                        if hue and hue_val is not None and param_dict.get(hue) != hue_val:
                            continue
                        if facet_by and facet_val is not None and param_dict.get(facet_by) != facet_val:
                            continue
                            
                        # Plot individual points with slight jitter
                        x_val = param_dict.get(x_axis)
                        if x_val is not None:
                            jitter = np.random.normal(0, 0.05, size=len(group_df))
                            ax.scatter(
                                [x_val + j for j in jitter], 
                                group_df[metric], 
                                alpha=0.3, 
                                color=color,
                                marker='x',
                                s=30
                            )
            
            # Set axis properties
            if x_axis in ['n_train', 'num_features']:
                ax.set_xscale('log')
            if metric.endswith('_mae') or metric.endswith('_loss'):
                ax.set_yscale('log')
                
            # Set titles and labels
            if facet_by and facet_val is not None:
                ax.set_title(f"{facet_by}={facet_val}")
            ax.set_xlabel(x_axis.replace('_', ' ').title())
            ax.set_ylabel(f"{metric.replace('_', ' ').title()}\n(mean ± std)")
            ax.grid(True, alpha=0.3, linestyle='--')
            
            # Add legend if we have hue values
            if hue and len(hue_values) > 1:
                ax.legend(title=hue.replace('_', ' ').title())
                
        # Add overall title
        fig.suptitle(f"{metric.replace('_', ' ').title()} vs {x_axis.replace('_', ' ').title()}\n(with standard deviation across runs)",
                    fontsize=16, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        # Save the plot
        plots_dir = Path("src/results") / self.df["experiment_name"].iloc[0] / "plots"
        plots_dir.mkdir(exist_ok=True, parents=True)
        save_path = plots_dir / f"error_bars_{metric}_{x_axis}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        self.logger.success(f"Saved error bar plot to {save_path}")
