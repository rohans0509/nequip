# src/experiments/training_manager.py
import subprocess
from pathlib import Path
from typing import Optional

class TrainingManager:
    def __init__(self, base_dir: str = "results"):
        self.base_dir = Path(base_dir)
    
    def train(self, config_path: str) -> None:
        """Run nequip training."""
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        cmd = f"nequip-train {config_path}"
        try:
            # Simple direct execution - output goes straight to terminal
            subprocess.run(cmd, shell=True, check=True)
                    
        except subprocess.CalledProcessError as e:
            print(f"Training failed with error code {e.returncode}")
            raise
        
    def deploy(self, train_dir: str, output_path: Optional[str] = None) -> None:
        """Deploy trained model."""
        train_dir = Path(train_dir)
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
            
        if output_path is None:
            output_path = train_dir / "deployed.pth"
        
        cmd = f"nequip-deploy build --train-dir {str(train_dir)} {str(output_path)}"
        try:
            subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"Model deployment failed with error:\n{e.stderr}")
            raise
    
    def evaluate(self, train_dir: str, batch_size: int = 50) -> None:
        """Evaluate model."""
        train_dir = Path(train_dir)
        if not train_dir.exists():
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
            
        log_file = train_dir / "test_results.txt"
        cmd = f"nequip-evaluate --train-dir {str(train_dir)} --batch-size {batch_size} --log {str(log_file)}"
        try:
            result = subprocess.run(
                cmd, 
                shell=True, 
                check=True,
                capture_output=True,
                text=True
            )
            print(f"Evaluation output:\n{result.stdout}")
        except subprocess.CalledProcessError as e:
            print(f"Evaluation failed with error:\n{e.stderr}")
            raise