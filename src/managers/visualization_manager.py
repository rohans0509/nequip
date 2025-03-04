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
from src.managers.logging_manager import LoggingManager
from src.managers.experiment_tracker import ExperimentTracker

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
