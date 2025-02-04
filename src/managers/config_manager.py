# src/utils/config_utils.py
import yaml
from pathlib import Path
from e3nn import o3
from typing import Dict, Any, List
from itertools import product
from src.settings import TOTAL_LAYERS
from src.managers.logging_manager import LoggingManager

def irreps(lvalue: int, num_features: int = 32, even: bool = False) -> str:
    """Generate irreps string for e3nn."""
    return str(o3.Irreps(
        [(num_features, (l, p))
         for p in ((1, -1) if not even else (1,))
         for l in range(lvalue + 1)]
    ))

class ConfigManager:
    def __init__(self, template_dir: str = "config_templates"):
        self.template_dir = Path(template_dir)
        self.logger = LoggingManager()
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from yaml file."""
        self.logger.info(f"Loading config from: {config_path}")
        
        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
                self.logger.log_dict(config, "Loaded Configuration")
                return config
            except yaml.YAMLError as exc:
                self.logger.error(f"Error loading config: {exc}")
                return {}

    def save_config(self, config: Dict[str, Any], save_path: str) -> None:
        """Save configuration to yaml file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as stream:
            try:
                yaml.dump(config, stream, default_flow_style=False)
            except yaml.YAMLError as exc:
                self.logger.error(f"Error saving config: {exc}")

    @staticmethod
    def update_config(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Update configuration with new parameters."""
        for key, value in kwargs.items():
            config[key] = value
        return config

    def generate_layer_irreps(self, lmax: int, num_features: int, inv_layers: int) -> str:
        """Generate layer irreps string."""
        params = {
            "lmax": lmax,
            "features": num_features,
            "inv_layers": inv_layers
        }
        self.logger.log_dict(params, "Layer Irreps Parameters")
        
        # If lmax is 0, the network is fully invariant.
        # Override inv_layers to be TOTAL_LAYERS.
        if lmax == 0:
            self.logger.info("lmax=0 detected, building fully invariant network: setting inv_layers to TOTAL_LAYERS.")
            inv_layers = TOTAL_LAYERS
            num_layers = 0
        else:
            num_layers = TOTAL_LAYERS - inv_layers
        
        layer_irreps = []
        if num_layers > 0:
            layer_irreps.extend([irreps(lvalue=lmax, num_features=num_features, even=False) 
                                 for _ in range(num_layers)])
        # Always add the invariant layers.
        layer_irreps.extend([irreps(lvalue=0, num_features=num_features, even=True) 
                             for _ in range(inv_layers)])
        
        result = ",".join(layer_irreps)
        self.logger.success("Successfully generated layer irreps")
        return result