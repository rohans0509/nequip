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
        """Main entry point for the dashboard"""
        st.set_page_config(
            page_title="Experiment Visualization Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        
        st.title("Experiment Visualization Dashboard")
        st.sidebar.title("Visualization Options")
        
        # Visualization type selector
        viz_type = st.sidebar.selectbox(
            "Select Visualization Type",
            ["Dataset Comparison", "Parameter Comparison", "Time Series", "Parameter Heatmap", "Raw Data Explorer"]
        )
        
        # Call the appropriate visualization method
        if viz_type == "Dataset Comparison":
            self.dataset_comparison_view()
        elif viz_type == "Parameter Comparison":
            self.parameter_comparison_view()
        elif viz_type == "Time Series":
            self.time_series_view()
        elif viz_type == "Parameter Heatmap":
            self.parameter_heatmap_view()
        elif viz_type == "Raw Data Explorer":
            self.raw_data_explorer()
    
    def get_common_filters(self):
        """Get common filter controls that appear in multiple visualization types"""
        st.sidebar.subheader("Filters")
        
        # Dataset filter
        datasets = ["All"] + self.viz.get_available_datasets()
        selected_dataset = st.sidebar.selectbox("Dataset", datasets)
        
        # Date range filter
        min_date, max_date = self.viz.get_date_range()
        if min_date and max_date:
            min_date = min_date.date()
            max_date = max_date.date()
            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            if len(date_range) == 2:
                start_date, end_date = date_range
                start_date = datetime.combine(start_date, datetime.min.time())
                end_date = datetime.combine(end_date, datetime.max.time())
            else:
                start_date, end_date = None, None
        else:
            start_date, end_date = None, None
        
        # Parameter filters
        lmax_values = ["All"] + [str(x) for x in sorted(self.df["lmax"].dropna().unique())]
        selected_lmax = st.sidebar.selectbox("L-max", lmax_values)
        
        inv_layers_values = ["All"] + [str(x) for x in sorted(self.df["inv_layers"].dropna().unique())]
        selected_inv_layers = st.sidebar.selectbox("Invariant Layers", inv_layers_values)
        
        # Build filter dict
        filters = {}
        if selected_dataset != "All":
            filters["dataset"] = selected_dataset
        if selected_lmax != "All":
            filters["lmax"] = int(selected_lmax)
        if selected_inv_layers != "All":
            filters["inv_layers"] = int(selected_inv_layers)
            
        return filters, start_date, end_date
    
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
        
        filters, start_date, end_date = self.get_common_filters()
        metric = st.selectbox("Select Metric", self.get_available_metrics())
        
        # Create filtered dataframe
        filtered_df = self.df.copy()
        
        # Add display name column
        filtered_df['display_name'] = filtered_df['experiment_id'].apply(self.get_friendly_experiment_id)
        
        for key, value in filters.items():
            filtered_df = filtered_df[filtered_df[key] == value]
            
        # Apply date filtering if needed
        if start_date and end_date:
            filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & 
                                     (filtered_df['timestamp'] <= end_date)]
        
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
        
        filters, start_date, end_date = self.get_common_filters()
        metric = self.get_available_metrics()
            
        if st.button("Generate Plot"):
            # Create filtered dataframe
            filtered_df = self.df.copy()
            for key, value in filters.items():
                filtered_df = filtered_df[filtered_df[key] == value]
            
            # Apply date filtering if needed
            if start_date and end_date:
                filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & 
                                         (filtered_df['timestamp'] <= end_date)]
            
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
        
        filters, start_date, end_date = self.get_common_filters()
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
            start_date_str = start_date.strftime("%Y-%m-%d") if start_date else None
            end_date_str = end_date.strftime("%Y-%m-%d") if end_date else None
            
            # Generate the plot
            viz.plot_metric_over_time(metric, group_by, start_date_str, end_date_str, rolling_window)
            
            # Find the saved plot
            plots_dir = Path("src/results") / self.df["experiment_name"].iloc[0] / "plots"
            date_suffix = ""
            if start_date_str or end_date_str:
                date_suffix = f"_{start_date_str or 'start'}_{end_date_str or 'end'}"
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
        
        filters, start_date, end_date = self.get_common_filters()
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
            if start_date and end_date:
                filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & 
                                         (filtered_df['timestamp'] <= end_date)]
            
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
        
        filters, start_date, end_date = self.get_common_filters()
        
        # Create filtered dataframe
        filtered_df = self.df.copy()
        for key, value in filters.items():
            filtered_df = filtered_df[filtered_df[key] == value]
        
        # Apply date filtering if needed
        if start_date and end_date:
            filtered_df = filtered_df[(filtered_df['timestamp'] >= start_date) & 
                                     (filtered_df['timestamp'] <= end_date)]
        
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

if __name__ == "__main__":
    dashboard = Dashboard()
    dashboard.run() 