# visualise.py
from src.managers.visualization_manager import VisualizationManager
import src.settings as settings

vm = VisualizationManager(
    experiment_name=settings.EXPERIMENT_NAME, 
    base_dir=settings.BASE_DIR
)

# vm.visualize_results()

# Generate all visualizations
# vm.plot_training_time_scaling()
# vm.plot_epoch_time_breakdown()
# vm.plot_computational_efficiency_heatmap()
# vm.plot_accuracy_vs_compute_tradeoff(metric='validation_f_mae')
vm.visualize_results()
# vm.plot_wall_time_comparison()