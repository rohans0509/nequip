# src/utils/config_utils.py
import yaml
from pathlib import Path
from e3nn import o3
from typing import Dict, Any, List
from itertools import product
from src.settings import TOTAL_LAYERS

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
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from yaml file."""
        with open(config_path, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(f"Error loading config: {exc}")
                return {}

    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str) -> None:
        """Save configuration to yaml file."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as stream:
            try:
                yaml.dump(config, stream, default_flow_style=False)
            except yaml.YAMLError as exc:
                print(f"Error saving config: {exc}")

    @staticmethod
    def update_config(config: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Update configuration with new parameters."""
        for key, value in kwargs.items():
            config[key] = value
        return config

    def generate_layer_irreps(self, lmax: int, num_features: int, inv_layers: int) -> str:
        """Generate layer irreps string."""
        num_layers = TOTAL_LAYERS - inv_layers
        layer_irreps = [irreps(lvalue=lmax, num_features=num_features, even=False) 
                       for _ in range(num_layers)]
        layer_irreps += [irreps(lvalue=0, num_features=num_features, even=True) 
                        for _ in range(inv_layers)]
        return ",".join(layer_irreps)