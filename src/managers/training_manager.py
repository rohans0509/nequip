# src/experiments/training_manager.py
import subprocess
from pathlib import Path
from typing import Optional
from src.managers.logging_manager import LoggingManager

class TrainingManager:
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
        self.logger = LoggingManager()
    
    def train(self, config_path: str) -> None:
        """Run nequip training."""
        self.logger.section("Training Process")
        
        if not Path(config_path).exists():
            self.logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        self.logger.info(f"Starting training with config: {config_path}")
        cmd = f"nequip-train {config_path}"
        try:
            self.logger.info("Training in progress...")
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            self.logger.success("Training completed successfully")
            
            if result.stdout:
                self.logger.section("Training Output")
                self.logger.info(result.stdout)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Training failed with error code {e.returncode}")
            if e.stdout:
                self.logger.section("Training Output Before Failure")
                self.logger.info(e.stdout)
            if e.stderr:
                self.logger.section("Error Output")
                self.logger.error(e.stderr)
            raise
        
        self.logger.divider()
    
    def deploy(self, train_dir: str, output_path: Optional[str] = None) -> None:
        """Deploy trained model."""
        train_dir = Path(train_dir)
        if not train_dir.exists():
            self.logger.error(f"Training directory not found: {train_dir}")
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
            
        if output_path is None:
            output_path = train_dir / "deployed.pth"
        
        self.logger.info(f"Deploying model from {train_dir} to {output_path}")
        cmd = f"nequip-deploy build --train-dir {str(train_dir)} {str(output_path)}"
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            self.logger.success("Model deployed successfully")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Model deployment failed:\n{e.stderr}")
            raise
    
    def evaluate(self, train_dir: str, batch_size: int = 50) -> None:
        """Evaluate model."""
        train_dir = Path(train_dir)
        if not train_dir.exists():
            self.logger.error(f"Training directory not found: {train_dir}")
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
            
        log_file = train_dir / "test_results.txt"
        self.logger.info(f"Evaluating model from {train_dir}")
        cmd = f"nequip-evaluate --train-dir {str(train_dir)} --batch-size {batch_size} --log {str(log_file)}"
        try:
            result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
            self.logger.success("Evaluation completed successfully")
            self.logger.info(f"Evaluation output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Evaluation failed:\n{e.stderr}")
            raise