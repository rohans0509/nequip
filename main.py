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

from src.managers.experiment_manager import ExperimentManager
from src.managers.training_manager import TrainingManager
from src.managers.visualization_manager import VisualizationManager
from src.managers.experiment_tracker import ExperimentTracker
import src.settings as settings
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def is_experiment_complete(config_path: str) -> bool:
    """
    Check whether an experiment is complete by verifying the existence of key output files.
    Returns True if both the deployed model and evaluation log exist.
    """
    config_file = Path(config_path)
    config_name = config_file.stem
    train_dir = config_file.parent.parent / config_name
    deployed_file = train_dir / "deployed.pth"
    eval_file = train_dir / "test_results.txt"
    return train_dir.exists() and deployed_file.exists() and eval_file.exists()

def main():
    # Initialize managers.
    experiment = ExperimentManager(experiment_name=settings.EXPERIMENT_NAME, base_dir=settings.BASE_DIR)
    trainer = TrainingManager()
    
    # Create a new experiment version
    version_dir = experiment.create_experiment_version()
    print(f"Using experiment directory: {version_dir}")
    
    # Generate configuration files from the hyperparameter grid (which now includes dataset).
    config_paths = experiment.generate_configs(
        param_grid=settings.PARAM_GRID,
        version_dir=version_dir
    )
    
    # Filter out experiments that are already complete.
    incomplete_configs = []
    for config_path in config_paths:
        config_name = Path(config_path).stem
        if is_experiment_complete(config_path):
            print(f"Skipping experiment '{config_name}' – experiment already complete.")
        else:
            incomplete_configs.append(config_path)
    
    print(f"Total experiments to run: {len(incomplete_configs)} out of {len(config_paths)}")
    print("\nConfigurations to be run:")
    for config_path in incomplete_configs:
        config_name = Path(config_path).stem
        print(f"- {config_name}")
    print()
    
    # Run experiments.
    for config_path in tqdm(incomplete_configs, desc="Running experiments"):
        config_name = Path(config_path).stem
        train_dir = Path(config_path).parent.parent / config_name
        print(f"Training with config: {config_path}")
        trainer.train(config_path)
        trainer.deploy(str(train_dir))
        trainer.evaluate(str(train_dir))
    
    # Update the centralized experiment log.
    tracker = ExperimentTracker()
    tracker.build_experiment_df()
    
    # Visualize results using the centralized DataFrame.
    vis_manager = VisualizationManager(df=tracker.df)
    vis_manager.plot_param_comparison(metric="final_validation_loss", fixed_params={})
    
if __name__ == "__main__":
    main()
