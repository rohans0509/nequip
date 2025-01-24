# src/experiments/experiment_manager.py
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from itertools import product
from src.managers.config_manager import ConfigManager
from src.managers.logging_manager import LoggingManager

class ExperimentManager:
    def __init__(self, 
                 experiment_name: str,
                 base_dir: str = "results"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.config_manager = ConfigManager()
        self.logger = LoggingManager()
        
    def create_experiment_version(self) -> str:
        """Create new experiment version directory."""
        timestamp = datetime.now().strftime("%d_%m_%Y_%I%M%p").lower()
        version_dir = self.base_dir / self.experiment_name / f"{timestamp}"
        
        self.logger.info(f"Creating new experiment version: {version_dir}")
        # Create directory structure
        for subdir in ['checkpoints', 'configs', 'logs', 'metrics']:
            (version_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        self.logger.success(f"Created experiment directory structure at {version_dir}")
        return str(version_dir)
    
    def generate_configs(self, base_config_path: str, param_grid: Dict[str, List], version_dir: str) -> List[str]:
        """Generate configurations for experiments."""
        self.logger.section("Configuration Generation")
        self.logger.info(f"Loading base config from {base_config_path}")
        
        base_config = self.config_manager.load_config(base_config_path)
        if not base_config:
            self.logger.error("Failed to load base config")
            return []
            
        config_paths = []
        total_combinations = len(list(product(*param_grid.values())))
        
        varying_params = {k: v for k, v in param_grid.items() if len(v) > 1}
        self.logger.log_dict(varying_params, "Varying Parameters")
        
        with self.logger.create_progress() as progress:
            task = progress.add_task(
                f"[cyan]Generating {total_combinations} configurations...", 
                total=total_combinations
            )
            
            keys, values = zip(*param_grid.items())
            for v in product(*values):
                params = dict(zip(keys, v))
                
                # Create run name using all parameters if none vary, otherwise use varying parameters
                if varying_params:
                    run_parts = [f"{k}_{params[k]}" for k in varying_params.keys()]
                else:
                    run_parts = ["default_run"]
                
                run_name = "_".join(run_parts)
                self.logger.info(f"Generating config for run: {run_name}")
                
                # Update config with experiment directory structure
                params['root'] = str(Path(version_dir))
                params['run_name'] = run_name
                
                # Generate layer irreps if needed
                if all(k in params for k in ['lmax', 'num_features', 'inv_layers']):
                    try:
                        layer_irreps = self.config_manager.generate_layer_irreps(
                            params['lmax'], 
                            params['num_features'], 
                            params['inv_layers']
                        )
                        params['layer_irreps'] = layer_irreps
                    except Exception as e:
                        self.logger.error(f"Failed to generate layer irreps for {run_name}: {e}")
                        continue
                
                # Update config
                config = self.config_manager.update_config(
                    base_config.copy(),
                    **params
                )
                
                # Save config
                config_path = Path(version_dir) / 'configs' / f"{run_name}.yaml"
                try:
                    self.config_manager.save_config(config, config_path)
                    config_paths.append(str(config_path))
                    self.logger.info(f"Saved config to {config_path}")
                except Exception as e:
                    self.logger.error(f"Failed to save config for {run_name}: {e}")
                
                progress.advance(task)
        
        self.logger.success(f"Generated {len(config_paths)} configurations successfully")
        if len(config_paths) < total_combinations:
            self.logger.warning(f"Failed to generate {total_combinations - len(config_paths)} configurations")
        
        self.logger.divider()
        return config_paths