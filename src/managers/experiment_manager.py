# src/managers/experiment_manager.py
"""
ExperimentManager module:
This module handles creating or selecting experiment version directories and generating configuration
files for each experiment run. It uses the hyperparameter grid (which now includes a dataset key) and
selects the appropriate base config from settings.BASE_CONFIGS.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from itertools import product
from src.managers.config_manager import ConfigManager
from src.managers.logging_manager import LoggingManager
from src.settings import TOTAL_LAYERS, BASE_CONFIGS

class ExperimentManager:
    def __init__(self, experiment_name: str, base_dir: str = "results"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.config_manager = ConfigManager()
        self.logger = LoggingManager()
        
    def create_experiment_version(self, existing_version: Optional[str] = None) -> str:
        """
        Create or use an existing experiment version directory.
        Ensures required subdirectories exist.
        """
        if existing_version:
            version_dir = Path(existing_version)
            self.logger.info(f"Using existing experiment version: {version_dir}")
            for subdir in ['checkpoints', 'configs', 'logs', 'metrics']:
                (version_dir / subdir).mkdir(parents=True, exist_ok=True)
        else:
            timestamp = datetime.now().strftime("%d_%m_%Y_%I%M%p").lower()
            version_dir = self.base_dir / self.experiment_name / f"{timestamp}"
            self.logger.info(f"Creating new experiment version: {version_dir}")
            for subdir in ['checkpoints', 'configs', 'logs', 'metrics']:
                (version_dir / subdir).mkdir(parents=True, exist_ok=True)
            self.logger.success(f"Created experiment directory structure at {version_dir}")
        
        return str(version_dir)
    
    def generate_configs(self, version_dir, param_grid):
        """
        Generate configuration files for all parameter combinations.
        
        Args:
            version_dir (str): Directory to save configurations.
            param_grid (dict): Dictionary of parameter grids.
            
        Returns:
            list: List of configuration file paths.
        """
        from itertools import product
        Path(version_dir).mkdir(parents=True, exist_ok=True)
        Path(version_dir, 'configs').mkdir(exist_ok=True)
        
        config_paths = []
        total_combinations = len(list(product(*param_grid.values())))
        
        varying_params = {k: v for k, v in param_grid.items() if len(v) > 1}
        self.logger.log_dict(varying_params, "Varying Parameters")
        
        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
        with self.logger.create_progress() as progress:
            task = progress.add_task(
                f"[cyan]Generating {total_combinations} configurations...", 
                total=total_combinations
            )
            
            keys, values = zip(*param_grid.items())
            for v in product(*values):
                params = dict(zip(keys, v))
                
                # Handle the special case of lmax=0: validate that inv_layers is TOTAL_LAYERS
                # or skip this combination
                if 'lmax' in params and params['lmax'] == 0:
                    if 'inv_layers' in params and params['inv_layers'] != TOTAL_LAYERS:
                        # Skip this invalid combination unless it was explicitly requested
                        # in the varying parameters (which would be unusual but supported)
                        if 'lmax' in varying_params and 'inv_layers' in varying_params:
                            # Check if this specific combination was explicitly requested
                            was_explicitly_requested = False
                            for combo in product(*varying_params.values()):
                                combo_dict = dict(zip(varying_params.keys(), combo))
                                if combo_dict.get('lmax') == 0 and combo_dict.get('inv_layers') == params['inv_layers']:
                                    was_explicitly_requested = True
                                    break
                                    
                            if not was_explicitly_requested:
                                self.logger.info(
                                    f"Skipping invalid config: lmax=0 requires inv_layers={TOTAL_LAYERS}, but found inv_layers={params.get('inv_layers')}."
                                )
                                progress.advance(task)
                                continue
                            else:
                                # This combination was explicitly requested in varying_params,
                                # so we'll adjust inv_layers to TOTAL_LAYERS as required
                                self.logger.warning(
                                    f"Found explicit request for lmax=0 with inv_layers={params['inv_layers']}. "
                                    f"Setting inv_layers={TOTAL_LAYERS} as required for lmax=0."
                                )
                                params['inv_layers'] = TOTAL_LAYERS
                        else:
                            # This combination was generated from the grid but not explicitly varied,
                            # so we'll skip it as invalid
                            self.logger.info(
                                f"Skipping invalid config: lmax=0 requires inv_layers={TOTAL_LAYERS}, but found inv_layers={params.get('inv_layers')}."
                            )
                            progress.advance(task)
                            continue
                
                # For lmax > 0, all inv_layers values are valid, but inv_layers=TOTAL_LAYERS is redundant
                # since it would be equivalent to having a non-invariant network 
                elif params.get('lmax', 0) > 0 and params.get('inv_layers', 0) == TOTAL_LAYERS:
                    self.logger.info(
                        f"Skipping redundant config: lmax={params.get('lmax')} with inv_layers={TOTAL_LAYERS} "
                        f"is redundant (equivalent to a non-invariant network)."
                    )
                    progress.advance(task)
                    continue

                # Build run name from varying parameters.
                run_parts = [f"{k}_{params[k]}" for k in varying_params.keys()]
                run_name = "_".join(run_parts)
                self.logger.info(f"Generating config for run: {run_name}")
                
                params['root'] = str(Path(version_dir))
                params['run_name'] = run_name
                
                # Select the appropriate base config based on dataset.
                dataset = params.get("dataset", "default")
                base_config_path = str(BASE_CONFIGS.get(dataset, ""))
                if not base_config_path:
                    self.logger.error(f"No base config defined for dataset {dataset}.")
                    progress.advance(task)
                    continue
                
                base_config = self.config_manager.load_config(base_config_path)
                if not base_config:
                    self.logger.error(f"Failed to load base config for dataset {dataset}.")
                    progress.advance(task)
                    continue
                
                # Generate layer irreps.
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
                        progress.advance(task)
                        continue
                # remove dataset from params but deepcopy so we don't modify the original
                params2 = params.copy()
                params2.pop('dataset')
                config = self.config_manager.update_config(
                    base_config.copy(),
                    **params2
                )
                
                config_path = Path(version_dir) / 'configs' / f"{run_name}.yaml"
                if not config_path.exists():
                    try:
                        self.config_manager.save_config(config, config_path)
                        self.logger.info(f"Saved config to {config_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to save config for {run_name}: {e}")
                        progress.advance(task)
                        continue
                else:
                    self.logger.info(f"Config already exists at {config_path}, skipping save.")
                
                config_paths.append(str(config_path))
                progress.advance(task)
        
        self.logger.success(f"Generated {len(config_paths)} configurations successfully")
        if len(config_paths) < total_combinations:
            self.logger.warning(f"Failed to generate {total_combinations - len(config_paths)} configurations")
        
        self.logger.divider()
        return config_paths
