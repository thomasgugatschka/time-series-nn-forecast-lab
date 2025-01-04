from ray import tune

from src.utilities.logging_utils import get_logger
logger = get_logger(__name__)

class HyperparameterManager:
    def __init__(self, config_hyperparameter):
        self.config_hyperparameter = config_hyperparameter

    def get_search_space(self, model_type):
        model_config = self.config_hyperparameter.get(model_type, None)
        if model_config is None:
            raise ValueError(f"Model type '{model_type}' not found in the hyperparameter configuration!")

        # Parse the JSON-like dictionary to generate Ray Tune search space
        def parse_value(key, value):
            if isinstance(value, list):
                # Handle list of lists case (e.g., TCN num_channels)
                if all(isinstance(item, list) for item in value):
                    # If all elements are lists, return as a categorical choice of list options
                    parsed_value = tune.choice(value)
                # Handle "log" indicator for log-uniform sampling
                elif len(value) == 3 and value[2] == "log":
                    parsed_value = tune.loguniform(value[0], value[1])
                # Handle range with two elements
                elif len(value) == 2:
                    if isinstance(value[0], int):
                        parsed_value = tune.randint(value[0], value[1] + 1)  # To include the upper limit
                    elif isinstance(value[0], float):
                        parsed_value = tune.uniform(value[0], value[1])
                    else:
                        parsed_value = tune.choice(value)  # Catch-all for other list values
                # Handle categorical lists
                else:
                    parsed_value = tune.choice(value)  # For categorical lists like ["Adam", "SGD"]
            else:
                parsed_value = value

            # Log each parsed value for debugging purposes
            logger.debug(f"Parsed value for key '{key}': {parsed_value}")
            return parsed_value

        # Generate the search space
        search_space_config = {
            "model_type": model_type,
            "model_params": {
                key: parse_value(key, val) for key, val in model_config.get("model_params", {}).items()
            },
            "training_params": {
                key: parse_value(key, val) for key, val in model_config.get("training_params", {}).items()
            },
            "trial_seed": tune.randint(0, 10000),  # Assign a different seed for each trial
        }

        # Debugging: Log the final search space
        logger.info(f"Search space for model type '{model_type}': {search_space_config}")
        return search_space_config


