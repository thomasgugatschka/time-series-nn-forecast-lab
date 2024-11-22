import logging
from ray import tune

# TODO -> REFACTOR so it is a bit nicer and cleaner and not a def within a def....

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
                # Check if the list has a "log" indicator for loguniform sampling
                if len(value) == 3 and value[2] == "log":
                    parsed_value = tune.loguniform(value[0], value[1])
                elif len(value) == 2:
                    # If list has two values, we assume it's a range for tuning
                    if isinstance(value[0], int):
                        parsed_value = tune.randint(value[0], value[1] + 1) #TODO is the +1 necessary???
                    elif isinstance(value[0], float):
                        parsed_value = tune.uniform(value[0], value[1])
                    else:
                        parsed_value = tune.choice(value)  # Catch-all for other list values
                else:
                    parsed_value = tune.choice(value)  # For categorical lists like ["Adam", "SGD"]
            else:
                parsed_value = value

            # Log each parsed value for debugging purposes
            logging.debug(f"Parsed value for key '{key}': {parsed_value}")
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
        logging.info(f"Search space for model type '{model_type}': {search_space_config}")
        return search_space_config
