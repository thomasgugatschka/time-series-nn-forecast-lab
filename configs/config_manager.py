import os
import yaml


class ConfigManager:
    def __init__(self, filepath):
        # Convert to an absolute path based on the project's root directory
        self.config = self.load_config(self.get_absolute_path(filepath))
        self.validate_config()

    @staticmethod
    def get_absolute_path(filepath):
        # Get the current directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Navigate up to the project root from the configs directory
        project_root = os.path.abspath(os.path.join(script_dir, '..'))  # Project root is one level up from configs

        # Construct the absolute path to the configuration file
        abs_path = os.path.join(project_root, filepath)
        return abs_path

    @staticmethod
    def load_config(filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found at {filepath}. Please check the file path.")

        with open(filepath, 'r') as f:
            return yaml.safe_load(f)

    def validate_config(self):
        # Example validation logic - can be extended
        required_keys = ['model_type', 'dataset', 'num_samples', 'cpu_per_trial', 'gpu_per_trial']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config[key]