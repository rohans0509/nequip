from experiments.experiment_manager import ExperimentManager
from experiments.config import ExperimentSettings

if __name__ == "__main__":
    settings = ExperimentSettings()
    print(settings)
    manager = ExperimentManager(settings)

    manager.run_experiments()