import argparse
from src.manager.config_manager import ConfigManager
from src.manager.path_manager import path_manager
from src.manager.file_manager import file_manager
from src.manager.experiment_manager import ExperimentManager
from src.utilities.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


# TODO -> REFACTOR -> Improve Performance, not needed to load everything. Maybe makes more sense to write hard-coded "validate_experiment.py" in root/scripts/
# Script should be used to quickly check if the relevant files are present after hyperparameter tuning or training

def parse_args():
    parser = argparse.ArgumentParser(description="Validate a single experiment using the metadata file.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Path to the configuration file to validate."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config_path = args.config_path

    config_manager = ConfigManager(args.config_path)

    # Initialize ExperimentManager
    experiment_manager = ExperimentManager(config_manager)

    # Compute hash for the given config file
    config_hash = experiment_manager._hash_config(config_manager.get_main_config_path())


    # Load metadata
    metadata = experiment_manager._load_experiment_metadata()

    # Check if hash exists in metadata
    if config_hash in metadata:
        logger.info(f" ### {config_path}: TRUE ###")
    else:
        logger.info(f" ### {config_path}: FALSE ###")


if __name__ == "__main__":
    main()
