from manager.path_manager import path_manager
from manager.file_manager import file_manager

class ConfigManager:
    def __init__(self, filepath):
        # Convert to an absolute path using PathManager and load configuration using FileManager
        # TODO maybe add something that absolute path is also allowed not only relative within the configs folder
        abs_path = path_manager.get_path_config() / filepath
        self.config = file_manager.load_yaml(abs_path)
        self.validate_config()

    def validate_config(self):
        # Updated validation logic to accommodate new keys and nested structures
        required_keys = ['model_type', 'dataset', 'num_samples', 'cpu_per_trial', 'gpu_per_trial',
                         'max_concurrent_trials', 'trial_params', 'train_val_test_split', 'search_algorithm', 'scheduler']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Missing required config key: {key}")

        # Validate trial_params keys
        trial_params_required_keys = ['max_epochs', 'grace_period', 'reduction_factor']
        trial_params = self.config.get('trial_params', {})
        for key in trial_params_required_keys:
            if key not in trial_params:
                raise ValueError(f"Missing required trial_params key: {key}")

        # Validate train_val_test_split keys
        split_required_keys = ['frac_train', 'frac_val']
        train_val_test_split = self.config.get('train_val_test_split', {})
        for key in split_required_keys:
            if key not in train_val_test_split:
                raise ValueError(f"Missing required train_val_test_split key: {key}")

    def get(self, key, default=None):
        return self.config.get(key, default)

    def __getitem__(self, key):
        return self.config[key]

# Create an instance of ConfigManager if needed, similar to how FileManager is handled
config_manager = ConfigManager('config.yaml')




# class ConfigManager:
#     def __init__(self, filepath):
#         # Convert to an absolute path based on the project's root directory
#         self.config = self.load_config(self.get_absolute_path(filepath))
#         self.validate_config()
#
#     @staticmethod
#     def get_absolute_path(filepath):
#         # Get the current directory where the script is located
#         script_dir = os.path.dirname(os.path.abspath(__file__))
#
#         # Navigate up to the project root from the configs directory
#         project_root = os.path.abspath(os.path.join(script_dir, '..'))  # Project root is one level up from configs
#
#         # Construct the absolute path to the configuration file
#         abs_path = os.path.join(project_root, filepath)
#         return abs_path
#
#     @staticmethod
#     def load_config(filepath):
#         if not os.path.exists(filepath):
#             raise FileNotFoundError(f"Config file not found at {filepath}. Please check the file path.")
#
#         with open(filepath, 'r') as f:
#             return yaml.safe_load(f)
#
#     # TODO - update this?!?!?!?!?!?!?!?!?!
#     def validate_config(self):
#         # Example validation logic - can be extended
#         required_keys = ['model_type', 'dataset', 'num_samples', 'cpu_per_trial', 'gpu_per_trial']
#         for key in required_keys:
#             if key not in self.config:
#                 raise ValueError(f"Missing required config key: {key}")
#
#     def get(self, key, default=None):
#         return self.config.get(key, default)
#
#     def __getitem__(self, key):
#         return self.config[key]