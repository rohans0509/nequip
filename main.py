# src/main.py
from src.managers.experiment_manager import ExperimentManager
from src.managers.training_manager import TrainingManager
from src.managers.visualization_manager import VisualizationManager
from pathlib import Path
import src.settings as settings
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

def is_experiment_complete(config_path: str) -> bool:
    """
    Check whether an experiment is complete.
    Considers an experiment complete if its training folder exists and contains both 
    a deployed model file ("deployed.pth") and evaluation output ("test_results.txt").
    """
    config_file = Path(config_path)
    config_name = config_file.stem
    # The training folder is assumed to be located at:
    # parent of 'configs' folder / <config_name>
    train_dir = config_file.parent.parent / config_name
    deployed_file = train_dir / "deployed.pth"
    eval_file = train_dir / "test_results.txt"
    return train_dir.exists() and deployed_file.exists() and eval_file.exists()

def main():
    # Initialize managers
    experiment = ExperimentManager(
        experiment_name=settings.EXPERIMENT_NAME,
        base_dir=settings.BASE_DIR
    )
    trainer = TrainingManager()
    
    # Specify the existing experiment folder to which you want to append new configurations
    existing_version = "src/results/aspirin_e3nn_study/28_01_2025_0816pm"
    version_dir = experiment.create_experiment_version(existing_version=existing_version)
    print(f"Using experiment directory: {version_dir}")
    
    # Generate configs -- new ones (e.g. for l=0) will be saved; existing ones will be reused.
    config_paths = experiment.generate_configs(
        base_config_path=str(settings.BASE_CONFIG_PATH),
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
    # List out the configs that will be run
    print("\nConfigurations to be run:")
    for config_path in incomplete_configs:
        config_name = Path(config_path).stem
        print(f"- {config_name}")
    print() # Empty line for readability
    
    # Loop over incomplete configurations with tqdm showing accurate count.
    for config_path in tqdm(incomplete_configs, desc="Running experiments"):
        config_name = Path(config_path).stem  # The name without extension
        train_dir = Path(config_path).parent.parent / config_name
        
        print(f"Training with config: {config_path}")
        trainer.train(config_path)
        
        # Deploy and evaluate the trained model
        trainer.deploy(str(train_dir))
        trainer.evaluate(str(train_dir))
    
    # Visualize results
    try:
        visualization_manager = VisualizationManager(
            experiment_name=settings.EXPERIMENT_NAME, 
            base_dir=settings.BASE_DIR
        )
        visualization_manager.visualize_results()
    except Exception as e:
        print(f"Visualization failed: {e}")
    
if __name__ == "__main__":
    main()