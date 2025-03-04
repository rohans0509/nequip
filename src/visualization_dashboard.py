#!/usr/bin/env python
# src/visualization_dashboard.py
"""
Streamlit dashboard for visualizing experiment results.
Run with: streamlit run src/visualization_dashboard.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
import io
import base64
import re

from src.managers.experiment_tracker import ExperimentTracker
from src.managers.visualization_manager import VisualizationManager
from src.managers.logging_manager import LoggingManager

class Dashboard:
    def __init__(self):
        self.logger = LoggingManager()
        self.tracker = ExperimentTracker()
        self.df = self.tracker.load_experiment_df()
        self.viz = VisualizationManager(self.df)
        
        # Convert timestamp to datetime if needed
        if 'timestamp' in self.df.columns and isinstance(self.df['timestamp'].iloc[0], str):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
    def get_friendly_experiment_id(self, experiment_id):
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
    
    def run(self):
        """Main entry point for the Streamlit dashboard"""
        st.set_page_config(
            page_title="NequIP Experiment Visualization",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        
        st.title("NequIP Experiment Visualization Dashboard")
        st.write("Interactive dashboard for visualizing experiment results")
        
        # Add navigation in sidebar
        st.sidebar.title("Navigation")
        page = st.sidebar.radio(
            "Select Page",
            ["Dataset Comparison", "Parameter Comparison", "Time Series", 
             "Parameter Heatmap", "Statistical Analysis", "Raw Data Explorer"]
        )
        
        # Navigate to the selected page
        if page == "Dataset Comparison":
            self.dataset_comparison_view()
        elif page == "Parameter Comparison":
            self.parameter_comparison_view()
        elif page == "Time Series":
            self.time_series_view()
        elif page == "Parameter Heatmap":
            self.parameter_heatmap_view()
        elif page == "Statistical Analysis":
            self.statistical_analysis_view()
        elif page == "Raw Data Explorer":
            self.raw_data_explorer()
            
    def get_common_filters(self):
        """
        Create a sidebar section for common filter options that apply to all visualizations.
        This includes dataset selection, date range, and parameter filters.
        
        Returns:
            Dict of filter parameters to apply to visualizations
        """
        st.sidebar.header("Common Filters")
        
        # Dataset selection
        available_datasets = ['All'] + self.df['dataset'].unique().tolist()
        selected_dataset = st.sidebar.selectbox("Dataset", available_datasets)
        
        # Date range selection
        date_min = self.df['timestamp'].min().date() if not self.df.empty else datetime.now().date()
        date_max = self.df['timestamp'].max().date() if not self.df.empty else datetime.now().date()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", 
                                      value=date_min,
                                      min_value=date_min,
                                      max_value=date_max)
        with col2:
            end_date = st.date_input("End Date", 
                                    value=date_max,
                                    min_value=date_min,
                                    max_value=date_max)
        
        # Parameters
        st.sidebar.subheader("Parameter Filters")
        
        # Lmax filter
        lmax_values = [None] + sorted([int(x) for x in self.df['lmax'].dropna().unique()])
        selected_lmax = st.sidebar.selectbox("L_max Value", lmax_values)
        
        # Invariant layers filter
        inv_layers_values = [None] + sorted([int(x) for x in self.df['inv_layers'].dropna().unique()])
        selected_inv_layers = st.sidebar.selectbox("Invariant Layers", inv_layers_values)
        
        # Test/Production filter
        is_test_options = ["All Experiments", "Production Only (is_test=False)", "Test Only (is_test=True)"]
        is_test_selection = st.sidebar.selectbox("Experiment Type", is_test_options)
        
        # Map selection to is_test value
        if is_test_selection == "Production Only (is_test=False)":
            is_test_value = False
        elif is_test_selection == "Test Only (is_test=True)":
            is_test_value = True
        else:
            is_test_value = None
        
        # Build filter dict
        filters = {}
        if selected_dataset != 'All':
            filters['dataset'] = selected_dataset
        if selected_lmax is not None:
            filters['lmax'] = selected_lmax
        if selected_inv_layers is not None:
            filters['inv_layers'] = selected_inv_layers
        if is_test_value is not None:
            filters['is_test'] = is_test_value
            
        # Apply date filter (handled separately)
        date_filter = (pd.to_datetime(start_date), pd.to_datetime(end_date))
        
        # Return both standard filters and date range
        return filters, date_filter
    
    def get_available_metrics(self):
        """Get available metrics for plotting"""
        metrics = self.viz.get_available_metrics()
        # Sort metrics by type for better organization
        metric_groups = {
            "Force MAE": [m for m in metrics if "f_mae" in m],
            "Energy MAE": [m for m in metrics if "e_mae" in m],
            "Loss": [m for m in metrics if "loss" in m],
            "Time": [m for m in metrics if "time" in m or "wall" in m],
            "Other": [m for m in metrics if not any(x in m for x in ["f_mae", "e_mae", "loss", "time", "wall"])]
        }
        
        # Create a hierarchical selectbox
        metric_category = st.sidebar.selectbox("Metric Category", list(metric_groups.keys()))
        return st.sidebar.selectbox("Metric", metric_groups[metric_category])
            
    def dataset_comparison_view(self):
        """View for comparing datasets"""
        st.header("Dataset Comparison")
        
        filters, date_filter = self.get_common_filters()
        metric = st.selectbox("Select Metric", self.get_available_metrics())
        
        # Create filtered dataframe
        filtered_df = self.df.copy()
        
        # Add display name column
        filtered_df['display_name'] = filtered_df['experiment_id'].apply(self.get_friendly_experiment_id)
        
        for key, value in filters.items():
            filtered_df = filtered_df[filtered_df[key] == value]
            
        # Apply date filtering if needed
        if date_filter:
            filtered_df = filtered_df[(filtered_df['timestamp'] >= date_filter[0]) & 
                                     (filtered_df['timestamp'] <= date_filter[1])]
        
        # UI controls for plot
        col1, col2 = st.columns(2)
        with col1:
            x_param = st.selectbox(
                "X-Axis Parameter", 
                ["n_train", "lmax", "inv_layers", "num_features"],
                index=0
            )
        with col2:
            hue_param = st.selectbox(
                "Color By", 
                ["dataset", "lmax", "inv_layers", "num_features"],
                index=0
            )
            
        facet_param = st.selectbox(
            "Facet By", 
            ["None", "dataset", "lmax", "inv_layers", "num_features"],
            index=0
        )
        
        # Convert "None" to None for the facet parameter
        facet_param = None if facet_param == "None" else facet_param
        
        if st.button("Generate Plot"):
            try:
                # Create a buffer to hold the plot image
                buf = io.BytesIO()
                
                viz = VisualizationManager(filtered_df)
                
                # Generate the plot
                viz.plot_by_dataset(metric, x_param, hue_param, facet_param, None)
                
                # Try to find the saved plot in the results directory
                try:
                    latest_plot = viz.get_latest_plot()
                    
                    if latest_plot:
                        # Display the plot
                        st.image(latest_plot)
                        
                        # Add download button
                        with open(latest_plot, "rb") as file:
                            btn = st.download_button(
                                label="Download Plot",
                                data=file,
                                file_name=f"{metric}_comparison.png",
                                mime="image/png"
                            )
                    else:
                        # Fallback: Check for plot in expected location based on metric
                        experiment_name = filtered_df["experiment_name"].iloc[0]
                        plots_dir = Path("src/results") / experiment_name / "plots"
                        expected_plot = plots_dir / f"dataset_comparison_{metric}.png"
                        
                        if expected_plot.exists():
                            # Display the plot
                            st.image(str(expected_plot))
                            
                            # Add download button
                            with open(expected_plot, "rb") as file:
                                btn = st.download_button(
                                    label="Download Plot",
                                    data=file,
                                    file_name=f"{metric}_comparison.png",
                                    mime="image/png"
                                )
                        else:
                            st.error("Plot file not found. Make sure plots directory exists and is writable.")
                except Exception as e:
                    st.error(f"Error locating plot file: {str(e)}")
            except Exception as e:
                st.error(f"Failed to generate plot: {str(e)}")
                st.info("Check that your data has sufficient points for the selected parameters and metric.")
    
    def parameter_comparison_view(self):
        """View for comparing how parameters affect metrics"""
        st.header("Parameter Comparison Visualization")
        
        filters, date_filter = self.get_common_filters()
        metric = self.get_available_metrics()
            
        if st.button("Generate Plot"):
            # Create filtered dataframe
            filtered_df = self.df.copy()
            for key, value in filters.items():
                filtered_df = filtered_df[filtered_df[key] == value]
            
            # Apply date filtering if needed
            if date_filter:
                filtered_df = filtered_df[(filtered_df['timestamp'] >= date_filter[0]) & 
                                         (filtered_df['timestamp'] <= date_filter[1])]
            
            viz = VisualizationManager(filtered_df)
            
            # Generate the plot
            viz.plot_param_comparison(metric, filters)
            
            # Find the saved plot
            plots_dir = Path("src/results") / self.df["experiment_name"].iloc[0] / "plots"
            plot_path = plots_dir / f"param_comparison_{metric}.png"
            
            if plot_path.exists():
                st.image(str(plot_path))
                
                # Add download button
                with open(plot_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Plot",
                        data=file,
                        file_name=f"param_comparison_{metric}.png",
                        mime="image/png"
                    )
            else:
                st.error("Plot generation failed or plot file not found")
    
    def time_series_view(self):
        """View for visualizing metric changes over time"""
        st.header("Time Series Visualization")
        
        filters, date_filter = self.get_common_filters()
        metric = self.get_available_metrics()
        
        group_by = st.selectbox("Group By", 
                               ["dataset", "lmax", "inv_layers", "status"],
                               index=0)
        
        rolling_window = st.slider("Rolling Average Window", 
                                  min_value=0, max_value=10, value=0,
                                  help="Number of experiments to average (0 for no averaging)")
        
        if rolling_window == 0:
            rolling_window = None
            
        if st.button("Generate Plot"):
            # Create filtered dataframe
            filtered_df = self.df.copy()
            for key, value in filters.items():
                filtered_df = filtered_df[filtered_df[key] == value]
            
            viz = VisualizationManager(filtered_df)
            
            # Convert dates to strings for the function
            date_filter_str = [f"{date.strftime('%Y-%m-%d')}" for date in date_filter]
            
            # Generate the plot
            viz.plot_metric_over_time(metric, group_by, date_filter_str[0], date_filter_str[1], rolling_window)
            
            # Find the saved plot
            plots_dir = Path("src/results") / self.df["experiment_name"].iloc[0] / "plots"
            date_suffix = ""
            if date_filter_str:
                date_suffix = f"_{date_filter_str[0]}_{date_filter_str[1]}"
            plot_path = plots_dir / f"time_series_{metric}{date_suffix}.png"
            
            if plot_path.exists():
                st.image(str(plot_path))
                
                # Add download button
                with open(plot_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Plot",
                        data=file,
                        file_name=f"time_series_{metric}{date_suffix}.png",
                        mime="image/png"
                    )
            else:
                st.error("Plot generation failed or plot file not found")
    
    def parameter_heatmap_view(self):
        """View for creating heatmaps of parameter interactions"""
        st.header("Parameter Heatmap Visualization")
        
        filters, date_filter = self.get_common_filters()
        metric = self.get_available_metrics()
        
        col1, col2 = st.columns(2)
        with col1:
            x_param = st.selectbox("X-Axis Parameter", 
                                  ["lmax", "inv_layers", "n_train", "num_features", "TOTAL_LAYERS"],
                                  index=0)
        with col2:
            y_param = st.selectbox("Y-Axis Parameter", 
                                  ["inv_layers", "lmax", "n_train", "num_features", "TOTAL_LAYERS"],
                                  index=0)
        
        if x_param == y_param:
            st.warning("X and Y parameters should be different for a meaningful heatmap")
            
        if st.button("Generate Plot"):
            # Create filtered dataframe
            filtered_df = self.df.copy()
            for key, value in filters.items():
                filtered_df = filtered_df[filtered_df[key] == value]
            
            # Apply date filtering if needed
            if date_filter:
                filtered_df = filtered_df[(filtered_df['timestamp'] >= date_filter[0]) & 
                                         (filtered_df['timestamp'] <= date_filter[1])]
            
            viz = VisualizationManager(filtered_df)
            
            # Generate the plot
            viz.plot_parameter_heatmap(metric, x_param, y_param, filters)
            
            # Find the saved plot
            plots_dir = Path("src/results") / self.df["experiment_name"].iloc[0] / "plots"
            filter_suffix = "_".join([f"{k}_{v}" for k, v in filters.items()]) if filters else ""
            if filter_suffix:
                filter_suffix = f"_{filter_suffix}"
            plot_path = plots_dir / f"heatmap_{metric}_{x_param}_{y_param}{filter_suffix}.png"
            
            if plot_path.exists():
                st.image(str(plot_path))
                
                # Add download button
                with open(plot_path, "rb") as file:
                    btn = st.download_button(
                        label="Download Plot",
                        data=file,
                        file_name=f"heatmap_{metric}_{x_param}_{y_param}{filter_suffix}.png",
                        mime="image/png"
                    )
            else:
                st.error("Plot generation failed or plot file not found")
    
    def raw_data_explorer(self):
        """View for exploring the raw experiment data"""
        st.header("Raw Data Explorer")
        
        filters, date_filter = self.get_common_filters()
        
        # Create filtered dataframe
        filtered_df = self.df.copy()
        for key, value in filters.items():
            filtered_df = filtered_df[filtered_df[key] == value]
        
        # Apply date filtering if needed
        if date_filter:
            filtered_df = filtered_df[(filtered_df['timestamp'] >= date_filter[0]) & 
                                     (filtered_df['timestamp'] <= date_filter[1])]
        
        # Add a display name column to make experiment IDs more readable
        filtered_df = filtered_df.copy()  # Create a copy to avoid SettingWithCopyWarning
        filtered_df['display_name'] = filtered_df['experiment_id'].apply(self.get_friendly_experiment_id)
        
        # Column selector
        all_columns = list(filtered_df.columns)
        default_columns = ["display_name", "dataset", "version", "timestamp", 
                          "lmax", "inv_layers", "n_train", 
                          "final_validation_f_mae", "final_validation_e_mae", "status"]
        default_indices = [all_columns.index(col) for col in default_columns if col in all_columns]
        
        selected_columns = st.multiselect(
            "Select Columns to Display",
            all_columns,
            default=[all_columns[i] for i in default_indices if i < len(all_columns)]
        )
        
        if not selected_columns:
            st.warning("Please select at least one column to display")
        else:
            # Display the data
            st.dataframe(filtered_df[selected_columns])
            
            # Add CSV download button
            csv = filtered_df[selected_columns].to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                "experiment_data.csv",
                "text/csv",
                key='download-csv'
            )

    def statistical_analysis_view(self):
        """
        View for statistical analysis of experiments across multiple runs.
        Generates plots with error bars showing mean ± std for metrics.
        """
        st.header("Statistical Analysis across Multiple Runs")
        st.write("""
        This view shows statistical analysis of experiments with the same configuration parameters.
        You can either view statistics with error bars across multiple runs or compare individual run iterations.
        """)
        
        # Get common filters
        filters, date_filter = self.get_common_filters()
        
        # Select metric
        metric = st.selectbox("Select Metric", self.get_available_metrics())
        
        # Plot configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            x_axis = st.selectbox("X-Axis Parameter", 
                                ['n_train', 'lmax', 'inv_layers', 'num_features', 'max_epochs'], 
                                index=0)
        with col2:
            hue = st.selectbox("Color By", 
                             ['lmax', 'inv_layers', 'dataset', 'None'], 
                             index=0)
            hue = None if hue == 'None' else hue
        with col3:
            facet_by = st.selectbox("Facet By", 
                                  ['inv_layers', 'lmax', 'dataset', 'None'], 
                                  index=0)
            facet_by = None if facet_by == 'None' else facet_by
        
        # View type selector
        view_type = st.radio(
            "View Type",
            ["Error Bars (Mean ± Std)", "Individual Run Iterations"]
        )
        
        # Advanced options in expander
        with st.expander("Advanced Grouping Options"):
            st.write("""
            By default, experiments are grouped by all parameters except those used for plotting.
            You can customize which parameters are used for grouping.
            """)
            
            # Let user select parameters to group by
            all_params = [col for col in self.df.columns if col not in 
                        ['experiment_id', 'timestamp', 'run_directory', 'metrics_file', 
                         'training_log_file', 'evaluation_log_file', 'deployed_model_file',
                         metric, 'final_training_loss', 'final_validation_loss']]
            
            use_custom_grouping = st.checkbox("Use custom grouping parameters", value=False)
            
            if use_custom_grouping:
                group_by = st.multiselect(
                    "Group By Parameters",
                    options=all_params,
                    default=['dataset', 'lmax', 'inv_layers', 'n_train', 'num_features']
                )
                
                exclude_params = st.multiselect(
                    "Exclude Parameters from Grouping",
                    options=all_params,
                    default=[]
                )
            else:
                group_by = None
                exclude_params = None
        
        # Generate button
        if st.button("Generate Statistical Analysis"):
            st.write("Generating statistical analysis plot...")
            
            # Create filtered dataframe
            filtered_df = self.df.copy()
            
            # Apply filters
            for key, value in filters.items():
                filtered_df = filtered_df[filtered_df[key] == value]
                
            # Apply date filtering
            if date_filter:
                filtered_df = filtered_df[(filtered_df['timestamp'] >= date_filter[0]) & 
                                         (filtered_df['timestamp'] <= date_filter[1])]
                
            if filtered_df.empty:
                st.warning("No data matches the selected filters")
                return
                
            # Create visualization manager with filtered data
            viz = VisualizationManager(filtered_df)
            
            if view_type == "Error Bars (Mean ± Std)":
                # Generate plot with error bars
                viz.plot_with_error_bars(
                    metric=metric,
                    x_axis=x_axis,
                    hue=hue,
                    facet_by=facet_by,
                    filters=None,  # Already filtered
                    group_by=group_by,
                    exclude_from_grouping=exclude_params
                )
                
                # Display the plot
                plots_dir = Path("src/results") / self.df["experiment_name"].iloc[0] / "plots"
                plot_path = plots_dir / f"error_bars_{metric}_{x_axis}.png"
                
                if plot_path.exists():
                    st.image(str(plot_path))
                    
                    # Add download button
                    with open(plot_path, "rb") as file:
                        btn = st.download_button(
                            label="Download Plot",
                            data=file,
                            file_name=f"error_bars_{metric}_{x_axis}.png",
                            mime="image/png"
                        )
                else:
                    st.error("Plot generation failed or file not found")
                    
                # Show detailed statistics
                st.subheader("Statistical Summary")
                
                # Create experiment groups
                experiment_groups = viz.group_by_config(group_by, exclude_params)
                
                # Calculate statistics
                stats_df = viz.calculate_group_statistics(experiment_groups, [metric])
                
                if not stats_df.empty:
                    # Reorder columns for better display
                    display_cols = ['metric', 'mean', 'std', 'min', 'max', 'count', 'cv']
                    param_cols = [c for c in stats_df.columns if c not in display_cols]
                    display_order = param_cols + display_cols
                    
                    # Show the statistics table
                    st.dataframe(stats_df[display_order])
                    
                    # Add CSV download
                    csv = stats_df.to_csv(index=False)
                    st.download_button(
                        label="Download Statistics as CSV",
                        data=csv,
                        file_name=f"statistics_{metric}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No statistical data available for the selected parameters")
                    
            else:  # Individual Run Iterations
                # Create figure
                plt.figure(figsize=(12, 8))
                
                # First, group experiments by configuration
                if 'run_iteration' not in exclude_params and exclude_params is not None:
                    exclude_params.append('run_iteration')
                elif exclude_params is None:
                    exclude_params = ['run_iteration']
                    
                # Group experiments by configuration (excluding run_iteration)
                experiment_groups = viz.group_by_config(group_by, exclude_params)
                
                # For each configuration, plot all run iterations
                for params, group_df in experiment_groups.items():
                    # Skip if only one run
                    if len(group_df) <= 1:
                        continue
                        
                    # Create a plot showing individual run iterations
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    # Get parameter values to set up plot title
                    param_dict = {p: v for p, v in params}
                    
                    # Extract values for x-axis and metric
                    x_values = group_df[x_axis].values
                    metric_values = group_df[metric].values
                    
                    # Get run iterations
                    run_iterations = group_df['run_iteration'].values
                    
                    # Plot individual points
                    for i, (x, y, run) in enumerate(zip(x_values, metric_values, run_iterations)):
                        ax.scatter(x, y, label=f"Run {run}", s=100, alpha=0.7)
                        ax.text(x, y, f"Run {run}", fontsize=9, ha='center', va='bottom')
                    
                    # Set axis labels and title
                    ax.set_xlabel(x_axis.replace('_', ' ').title())
                    ax.set_ylabel(metric.replace('_', ' ').title())
                    ax.set_title(f"{metric.replace('_', ' ').title()} for Multiple Runs\n" + 
                               ", ".join([f"{p}={v}" for p, v in params]))
                    
                    # Set log scale if appropriate
                    if x_axis in ['n_train', 'num_features']:
                        ax.set_xscale('log')
                    if metric.endswith('_mae') or metric.endswith('_loss'):
                        ax.set_yscale('log')
                    
                    # Add grid
                    ax.grid(True, alpha=0.3, linestyle='--')
                    
                    # Show the plot
                    st.pyplot(fig)
                    plt.close(fig)
                
                if not experiment_groups:
                    st.warning("No configurations with multiple run iterations found.")
                elif all(len(group_df) <= 1 for _, group_df in experiment_groups.items()):
                    st.warning("No configurations have multiple run iterations for comparison.")

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run() 