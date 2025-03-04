# src/main.py
"""
Main execution script:
This script orchestrates the entire workflow:
  - It creates or uses an existing experiment version.
  - Generates configuration files from the hyperparameter grid (supporting multiple datasets).
  - Runs training, deployment, and evaluation for incomplete experiments.
  - Updates the centralized experiment log.
  - Invokes visualization functions using the central DataFrame.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import time
import functools
import random
import argparse
import torch

from src.managers.experiment_manager import ExperimentManager
from src.managers.training_manager import TrainingManager
from src.managers.visualization_manager import VisualizationManager
from src.managers.experiment_tracker import ExperimentTracker
from src.managers.logging_manager import LoggingManager
import src.settings as settings
from src.utils.parse_helpers import parse_run_name, create_run_name

load_dotenv()
logger = LoggingManager()

# Use lru_cache to cache experiment completion status
@functools.lru_cache(maxsize=1024)
def is_experiment_complete(config_path: str) -> bool:
    """
    Check whether an experiment is complete by verifying the existence of key output files.
    Returns True if both the deployed model and evaluation log exist.
    
    This function is cached to avoid repeated filesystem checks.
    """
    config_file = Path(config_path)
    config_name = config_file.stem
    train_dir = config_file.parent.parent / config_name
    deployed_file = train_dir / "deployed.pth"
    eval_log = train_dir / "evaluation_log.json"
    
    return deployed_file.exists() and eval_log.exists()

def run_experiment(config_path: str) -> bool:
    """
    Run a single experiment with the given config path.
    Returns True if the experiment completed successfully.
    """
    try:
        config_file = Path(config_path)
        
        # For GPU-based training, it's important to initialize CUDA in each process
        # We set the device here, but the training manager will manage it
        os.environ["CUDA_VISIBLE_DEVICES"] = str(random.randint(0, torch.cuda.device_count() - 1) 
                                               if torch.cuda.device_count() > 1 else 0)
        
        # Skip if already complete
        if is_experiment_complete(config_path):
            logger.info(f"Skipping completed experiment: {config_file.stem}")
            return True
            
        # Create a training manager and run the experiment
        training_manager = TrainingManager(config_file)
        training_manager.train()
        training_manager.deploy()
        training_manager.evaluate()
        
        # Verify completion
        if is_experiment_complete(config_path):
            logger.success(f"Successfully completed experiment: {config_file.stem}")
            return True
        else:
            logger.error(f"Experiment did not complete successfully: {config_file.stem}")
            return False
    
    except Exception as e:
        logger.error(f"Error in experiment {config_path}: {str(e)}")
        return False

def parse_args():
    parser = argparse.ArgumentParser(description="Run NequIP experiments")
    parser.add_argument("--parallel", action="store_true", 
                       help="Run experiments in parallel (overrides settings.PARALLEL_RUNS)")
    parser.add_argument("--sequential", action="store_true", 
                       help="Run experiments sequentially (overrides settings.PARALLEL_RUNS)")
    parser.add_argument("--max-workers", type=int, 
                       help="Maximum number of parallel workers (overrides settings.MAX_WORKERS)")
    parser.add_argument("--num-runs", type=int, 
                       help="Number of runs per configuration (overrides settings.NUM_RUNS)")
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Initialize managers.
    logger.section("Initializing Experiment")
    experiment_manager = ExperimentManager()
    tracker = ExperimentTracker()
    
    # Override settings with command line args if provided
    parallel_runs = args.parallel if args.parallel else (not args.sequential if args.sequential else settings.PARALLEL_RUNS)
    max_workers = args.max_workers if args.max_workers else settings.MAX_WORKERS
    num_runs = args.num_runs if args.num_runs else settings.NUM_RUNS
    
    # Generate configurations
    logger.section("Generating Configurations")
    configs = []
    
    # For each parameter configuration
    for config_path in experiment_manager.get_config_files():
        config_file = Path(config_path)
        base_stem = config_file.stem
        
        # Parse the existing configuration name
        params = parse_run_name(base_stem)
        
        # For each run of this configuration
        for run_idx in range(num_runs):
            # Add run iteration to parameters
            run_params = params.copy()
            run_params['run'] = settings.RUN_LABELS[run_idx]
            
            # Create a new config name with run iteration
            run_name = create_run_name(run_params)
            
            # Generate the new config path
            run_config_path = config_file.parent / f"{run_name}.yaml"
            
            # If the config doesn't exist yet, create it by copying the base config
            if not run_config_path.exists():
                # Copy the base config to the new path
                with open(config_path, 'r') as src, open(run_config_path, 'w') as dst:
                    for line in src:
                        # Update the run name in the config
                        if line.strip().startswith('run_name:'):
                            dst.write(f'run_name: "{run_name}"\n')
                        # Add the run iteration parameter
                        elif line.strip() == 'is_test:' and 'run_iteration' not in line:
                            dst.write(line)
                            dst.write(f'run_iteration: {run_idx + 1}\n')
                        else:
                            dst.write(line)
                            
                logger.info(f"Created run configuration {run_name}")
            
            # Add to the list of configs to run
            configs.append(str(run_config_path))
    
    # Run experiments
    logger.section(f"Running {len(configs)} Experiments")
    
    # Shuffle configs to distribute workload more evenly
    random.shuffle(configs)
    
    if parallel_runs and len(configs) > 1:
        logger.info(f"Running experiments in parallel with {max_workers} workers")
        
        # Run in parallel with ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(run_experiment, configs))
            
        logger.info(f"Completed {sum(results)} out of {len(configs)} experiments")
    else:
        logger.info("Running experiments sequentially")
        
        # Run sequentially 
        completed = 0
        for config in configs:
            if run_experiment(config):
                completed += 1
                
        logger.info(f"Completed {completed} out of {len(configs)} experiments")
    
    # Update experiment tracking database
    logger.section("Updating Experiment Log")
    tracker.update_experiment_log()
    
    # Generate visualizations
    logger.section("Generating Visualizations")
    
    # Load the updated experiment DataFrame
    df = tracker.load_experiment_df()
    viz = VisualizationManager(df)
    
    # Generate standard visualizations with different metrics
    for metric in ['final_validation_f_mae', 'final_validation_e_mae', 'wall_time']:
        # Parameter comparison
        viz.plot_param_comparison(metric)
        
        # Dataset comparison
        viz.plot_by_dataset(metric, x_axis='n_train', hue='lmax', facet_by='inv_layers')
        
        # Error bar plots (only if we have multiple runs)
        if num_runs > 1:
            viz.plot_with_error_bars(metric, x_axis='n_train', hue='lmax', facet_by='inv_layers')
    
    logger.success("Experiment execution completed successfully")

if __name__ == "__main__":
    main()
