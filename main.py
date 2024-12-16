# src/main.py
from src.experiments.experiment_manager import ExperimentManager
from src.experiments.training_manager import TrainingManager
from pathlib import Path
import src.settings as settings

def main():
    # Initialize managers
    experiment = ExperimentManager(
        experiment_name=settings.EXPERIMENT_NAME,
        base_dir=settings.BASE_DIR
    )
    trainer = TrainingManager()
    
    # Create new experiment version
    version_dir = experiment.create_experiment_version()
    print(f"Created experiment at: {version_dir}")
    
    # Generate configs using settings
    config_paths = experiment.generate_configs(
        base_config_path=settings.BASE_CONFIG_PATH,
        param_grid=settings.PARAM_GRID,
        version_dir=version_dir
    )
    
    # Run experiments
    for config_path in config_paths:
        print(f"Training with config: {config_path}")
        trainer.train(config_path)
        
        # Get train directory from config path
        train_dir = Path(config_path).parent.parent / 'checkpoints'
        
        # Deploy and evaluate
        trainer.deploy(str(train_dir))
        trainer.evaluate(str(train_dir))

if __name__ == "__main__":
    main()