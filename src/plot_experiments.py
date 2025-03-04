#!/usr/bin/env python
# src/plot_experiments.py
"""
Command-line tool for visualizing experiment results.
This script provides a simple interface to the VisualizationManager.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from src.managers.experiment_tracker import ExperimentTracker
from src.managers.visualization_manager import VisualizationManager
from src.managers.logging_manager import LoggingManager

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize experiment results")
    
    # Main command structure
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Common arguments for all plot types
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--metric', type=str, required=True, 
                            help='Metric to visualize (e.g., final_validation_f_mae)')
    base_parser.add_argument('--dataset', type=str, help='Filter by dataset')
    base_parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    base_parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    base_parser.add_argument('--lmax', type=int, help='Filter by lmax value')
    base_parser.add_argument('--inv-layers', type=int, help='Filter by inv_layers value')
    
    # Param comparison command
    param_parser = subparsers.add_parser('param-comparison', parents=[base_parser],
                                       help='Plot parameter comparison')
    
    # Dataset comparison command
    dataset_parser = subparsers.add_parser('dataset-comparison', parents=[base_parser],
                                        help='Plot comparison across datasets')
    dataset_parser.add_argument('--x-axis', type=str, default='n_train',
                              help='Parameter to use for x-axis')
    dataset_parser.add_argument('--hue', type=str, default='lmax',
                              help='Parameter to use for color coding')
    dataset_parser.add_argument('--facet-by', type=str, default='inv_layers',
                              help='Parameter to use for faceting')
    
    # Time series command
    time_parser = subparsers.add_parser('time-series', parents=[base_parser],
                                       help='Plot metric changes over time')
    time_parser.add_argument('--group-by', type=str, default='dataset',
                           help='Parameter to group by in time series')
    time_parser.add_argument('--rolling-window', type=int,
                           help='Window size for rolling average')
    
    # Heatmap command
    heatmap_parser = subparsers.add_parser('heatmap', parents=[base_parser],
                                         help='Create parameter heatmap')
    heatmap_parser.add_argument('--x-param', type=str, required=True,
                              help='Parameter for x-axis of heatmap')
    heatmap_parser.add_argument('--y-param', type=str, required=True,
                              help='Parameter for y-axis of heatmap')
    
    # Info command to list available datasets and metrics
    info_parser = subparsers.add_parser('info', help='Show available datasets and metrics')
    
    return parser.parse_args()

def main():
    args = parse_args()
    logger = LoggingManager()
    
    # Load the experiment data
    tracker = ExperimentTracker()
    df = tracker.load_experiment_df()
    viz = VisualizationManager(df)
    
    # Handle the "info" command to list available options
    if args.command == 'info':
        datasets = viz.get_available_datasets()
        metrics = viz.get_available_metrics()
        start_date, end_date = viz.get_date_range()
        
        logger.section("Available Dataset Information")
        logger.info(f"Datasets: {', '.join(datasets)}")
        logger.info(f"Metrics: {', '.join(metrics)}")
        if start_date and end_date:
            logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        return
    
    # Check if a command was provided
    if not args.command:
        logger.error("No command specified. Use --help for usage information.")
        return
    
    # Build filter params from common arguments
    filters = {}
    if args.dataset:
        filters['dataset'] = args.dataset
    if args.lmax is not None:
        filters['lmax'] = args.lmax
    if args.inv_layers is not None:
        filters['inv_layers'] = args.inv_layers
    
    # Handle each command
    if args.command == 'param-comparison':
        viz.plot_param_comparison(args.metric, filters)
        
    elif args.command == 'dataset-comparison':
        date_range = None
        if args.start_date or args.end_date:
            date_range = (args.start_date, args.end_date)
        viz.plot_by_dataset(args.metric, args.x_axis, args.hue, args.facet_by, date_range)
        
    elif args.command == 'time-series':
        viz.plot_metric_over_time(
            args.metric, 
            args.group_by, 
            args.start_date, 
            args.end_date, 
            args.rolling_window
        )
        
    elif args.command == 'heatmap':
        viz.plot_parameter_heatmap(args.metric, args.x_param, args.y_param, filters)
    
    logger.success(f"Plot generation completed for command: {args.command}")

if __name__ == "__main__":
    main() 