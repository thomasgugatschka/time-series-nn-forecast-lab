import os

from manager.path_manager import path_manager
from manager.file_manager import file_manager
from utilities.logging_utils import get_logger

logger = get_logger(__name__)

class ConfigManager:
    def __init__(self, main_config_file='config.yaml'):
        # Load main configuration
        self.main_config_file_path = path_manager.get_path_config() / main_config_file
        self.main_config = file_manager.load_yaml(self.main_config_file_path)

        # Load additional configs (dataset-specific and model-specific)
        self.dataset_config = None
        self._load_additional_configs()

        # Perform configuration validation
        self.validate_config(self.main_config, "main_config")
        self.validate_config(self.dataset_config, "dataset_config")

    def _load_config(self, filepath):
        """Load a configuration file using the FileManager"""
        if os.path.isabs(filepath):
            abs_path = filepath
        else:
            abs_path = path_manager.get_path_config() / filepath
        logger.info(f"Loading configuration from {abs_path}")
        return file_manager.load_yaml(abs_path)

    def _load_additional_configs(self):
        # Load dataset-specific config
        dataset_name = self.main_config.get('dataset')
        if dataset_name:
            dataset_config_path = path_manager.get_path_dataset_config(dataset_name=dataset_name)
            self.dataset_config = self._load_config(dataset_config_path)
            logger.info(f"Loaded dataset-specific config for dataset: {dataset_name}")

        # Load more config files here if needed

    def validate_config(self, config, config_name):
        """Validate configuration based on config_name (main_config, dataset_config, data_processing_config)"""
        if config is None:
            logger.warning(f"No configuration found for {config_name}. Skipping validation.")
            return

        validation_rules = {
            "main_config": [
                ('model_type', True),
                ('dataset', True),
                ('num_samples', True),
                ('cpu_per_trial', True),
                ('gpu_per_trial', True),
                ('max_concurrent_trials', True),
                ('trial_params.max_epochs', True),
                ('trial_params.grace_period', True),
                ('trial_params.reduction_factor', True),
                ('data_params.data_split.frac_train', True),
                ('data_params.data_split.frac_val', True),
                ('data_params.data_split.frac_test', False), # Optional
                ('data_params.time_series_window', False), # Optional
                ('search_algorithm', True),
                ('scheduler', True)
            ],
            "dataset_config": [
                ('sources', True),
                ('feature_list', True),
                ('ts_past', True),
                ('ts_future', True),
                ('crop_start', False),
                ('crop_end', False),
                ('columns_to_drop', False)
            ]
        }

        rules = validation_rules.get(config_name, [])
        for rule in rules:
            key, required = rule
            if not self._validate_key(config, key) and required:
                raise ValueError(f"Missing required key '{key}' in {config_name}")

    def _validate_key(self, config, key):
        """Check if a nested key exists in a config dictionary"""
        keys = key.split('.')
        current = config
        for k in keys:
            if k in current:
                current = current[k]
            else:
                return False
        return True

    def get(self, key, default=None):
        return self.main_config.get(key, default)

    def get_dataset_config(self):
        return self.dataset_config

    def get_main_config(self):
        """Return the main configuration."""
        return self.main_config

    def get_main_config_path(self):
        """Get the absolute path of the main configuration file."""
        return str(self.main_config_file_path)

    def __getitem__(self, key):
        return self.main_config[key]


#config_manager = ConfigManager('config.yaml')
