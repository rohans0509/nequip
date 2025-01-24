# visualise.py
from src.managers.visualization_manager import VisualizationManager
import src.settings as settings

vm = VisualizationManager(
    experiment_name=settings.EXPERIMENT_NAME, 
    base_dir=settings.BASE_DIR
)


# Plot learning curves for n_train=2
vm.visualize_results()