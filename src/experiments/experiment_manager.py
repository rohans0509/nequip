# src/experiments/experiment_manager.py
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from itertools import product
from src.utils.config_utils import ConfigManager

class ExperimentManager:
    def __init__(self, 
                 experiment_name: str,
                 base_dir: str = "results"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.config_manager = ConfigManager()
        
    def create_experiment_version(self) -> str:
        """Create new experiment version directory."""
        timestamp = datetime.now().strftime("%d_%m_%Y_%I%M%p").lower()
        version_dir = self.base_dir / self.experiment_name / f"{timestamp}"
        
        # Create directory structure
        for subdir in ['checkpoints', 'configs', 'logs', 'metrics']:
            (version_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        return str(version_dir)
    
    def generate_configs(self, 
                        base_config_path: str,
                        param_grid: Dict[str, List],
                        version_dir: str) -> List[str]:
        """Generate configurations for experiments."""
        base_config = self.config_manager.load_config(base_config_path)
        config_paths = []
        
        # Find parameters that vary (have more than one value)
        varying_params = {k: v for k, v in param_grid.items() if len(v) > 1}
        
        # Generate all combinations of parameters
        keys, values = zip(*param_grid.items())
        for v in product(*values):
            params = dict(zip(keys, v))
            
            # Create run name using all parameters if none vary, otherwise use varying parameters
            if varying_params:
                run_parts = [f"{k}_{params[k]}" for k in varying_params.keys()]
            else:
                # Use a default name or some key parameters if no parameters vary
                run_parts = ["default_run"]  # or use some key parameters like:
                # run_parts = [f"lmax_{params['lmax']}_features_{params['num_features']}"]
            
            run_name = "_".join(run_parts)
            
            # Update config with experiment directory structure
            params['root'] = str(Path(version_dir))
            params['run_name'] = run_name
            
            # Generate layer irreps if needed
            if all(k in params for k in ['lmax', 'num_features', 'inv_layers']):
                layer_irreps = self.config_manager.generate_layer_irreps(
                    params['lmax'], 
                    params['num_features'], 
                    params['inv_layers']
                )
                params['layer_irreps'] = layer_irreps
            
            # Update config
            config = self.config_manager.update_config(
                base_config.copy(),
                **params
            )
            
            # Save config
            config_path = Path(version_dir) / 'configs' / f"{run_name}.yaml"
            self.config_manager.save_config(config, config_path)
            config_paths.append(str(config_path))
            
        return config_paths