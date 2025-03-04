# Experiment Visualization Tools

This directory contains tools for visualizing experiment results from the centralized CSV data storage.

## Prerequisites

Install the required dependencies:

```bash
pip install -r visualization_requirements.txt
```

## Available Visualization Tools

### 1. Command Line Interface

The `plot_experiments.py` script provides a powerful command-line interface for quickly generating plots:

```bash
# Show available datasets and metrics
python src/plot_experiments.py info

# Generate a parameter comparison plot
python src/plot_experiments.py param-comparison --metric final_validation_f_mae --dataset aspirin

# Compare datasets
python src/plot_experiments.py dataset-comparison --metric final_validation_f_mae --x-axis n_train --hue lmax

# Plot metric changes over time
python src/plot_experiments.py time-series --metric final_validation_f_mae --group-by lmax --start-date 2023-01-01 --end-date 2023-12-31

# Create a parameter heatmap
python src/plot_experiments.py heatmap --metric final_validation_f_mae --x-param lmax --y-param inv_layers --dataset aspirin
```

### 2. Interactive Dashboard

The `visualization_dashboard.py` script provides an interactive web-based dashboard for exploring experiment results:

```bash
streamlit run src/visualization_dashboard.py
```

The dashboard includes:
- Dataset comparison visualizations
- Parameter comparison plots
- Time series analysis
- Parameter heatmaps
- Raw data explorer with filtering and CSV export

## Visualization Features

### Filtering Options
- By dataset
- By date range
- By model parameters (lmax, inv_layers, etc.)

### Plot Types
1. **Parameter Comparison**: Shows how metrics change with training size, faceted by invariant layers
2. **Dataset Comparison**: Compares metrics across different datasets
3. **Time Series**: Tracks how metrics change over time with optional rolling averages
4. **Parameter Heatmap**: Shows how two parameters jointly affect a metric

### Additional Features
- Publication-quality plots with customizable appearance
- Log-scale handling for appropriate metrics
- Automatic regression line fitting
- Plot downloads in PNG format
- Raw data export in CSV format

## Examples

### Comparing Force MAE across datasets with different lmax values
```bash
python src/plot_experiments.py dataset-comparison --metric final_validation_f_mae --hue lmax
```

### Tracking validation loss over time 
```bash
python src/plot_experiments.py time-series --metric final_validation_loss --group-by dataset --rolling-window 3
```

### Creating a heatmap of lmax vs inv_layers impact on energy MAE
```bash
python src/plot_experiments.py heatmap --metric final_validation_e_mae --x-param lmax --y-param inv_layers
``` 