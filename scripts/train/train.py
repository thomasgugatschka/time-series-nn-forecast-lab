import argparse
import torch
import time
import logging
import ray
import json

from ray import tune
from ray.tune.experiment.trial import Trial
from ray.air import session
from sympy import false

from configs.logging_config import setup_logging
from configs.paths import get_path

from utilities.device_utils import get_device
from utilities.seed_utils import set_seed
from utilities.save_utils import save_parameters_and_results
from utilities.data_utils import preprocess_data, split_and_convert_data, z_score_denormalize, load_config
from utilities.plot_utils import plot_training_progress, plot_prediction, plot_prediction_with_context, create_gif_from_plots

from models.model_factory import model_factory
from models.wrapper_factory import wrapper_factory


# Load the configuration file
# TODO -> refactor naming.... so all config etc. have same same naming convention
# TODO REFACOTR -> so Load Config or so can be used... myaybe new util for files / file loading ?
with open(get_path('config_dir', 'hyperparameter_config.json'), "r") as f:
    config_hyperparameter = json.load(f)


import logging
from ray import tune

# Define the search space function based on loaded config
def search_space(model_type):
    model_config = config_hyperparameter.get(model_type, None)
    if model_config is None:
        raise ValueError(f"Model type {model_type} not found in the configuration!")

    def parse_value(key, value):
        if isinstance(value, list):
            if len(value) == 3 and value[2] == "log":
                parsed_value = tune.loguniform(value[0], value[1])
            elif len(value) == 2:
                if isinstance(value[0], int):
                    parsed_value = tune.randint(value[0], value[1])
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

    search_space_config = {
        "model_type": model_type,
        "model_params": {
            key: parse_value(key, val) for key, val in model_config.get("model_params", {}).items()
        },
        "training_params": {
            key: parse_value(key, val) for key, val in model_config.get("training_params", {}).items()
        },
    }

    # Debugging: Log the final search space
    logging.info(f"Search space for model type {model_type}: {search_space_config}")
    return search_space_config



# region ### PARSE ARGUMENTS ###
def parse_args():
    parser = argparse.ArgumentParser(description="Train different models with different datasets.")
    parser.add_argument("--model_type", type=str, required=True, help="Model type to train (RNN, LSTM, GRU, Transformer, CNN)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., sinusoidal, weather_station)")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of trials for hyperparameter tuning")
    parser.add_argument("--cpu_per_trial", type=int, default=2, help="Number of CPUs per trial")
    parser.add_argument("--gpu_per_trial", type=float, default=0.5, help="Number of GPUs per trial")
    parser.add_argument("--manual_hyperparams", action='store_true', help="Use manual hyperparameters instead of tuning")
    return parser.parse_args()
# endregion

def load_manual_hyperparameters():
    # TODO -> refactor so it is reading in the TRAIN_CONFIG FILES -> rework TRAINING CONFIG FILES
    # TODO -> BATCH SIZE ???
    return {
        "model_type": "RNN",
        "model_params": {
            "output_size": 10,
            "hidden_size": 32,
            "num_layers": 2,
            "dropout": 0.2,
        },
        "training_params": {
            "learning_rate": 0.005,
            "epochs": 50,
            "optimizer": "Adam",
            "loss_function": "MSELoss",
        }
    }

# region ### TRAINING FUNCTION ##
# --> train_model method
def objective(config, config_data, model_type):
    # Model-specific training process goes here
    set_seed(42)
    device = get_device()
    logging.info(f"Device: {device}")

    # region ### LOAD and PREPROCESS DATA ###
    df, x_arr, y_arr, z_mean, z_std, selected_feature, num_features = preprocess_data(config_data, dataset)
    # endregion

    # region ### SPLIT and CONVERT DATA ###
    # tODO are the remaining parameters "arr" not needed anywhere?
    x_train, x_val, x_test, y_train, y_val, y_test, _, _, _, _, _, _ = split_and_convert_data(x_arr, y_arr)
    # endregion

    # region ### SET VARIABLES from CONFIG ###
    model_params = config["model_params"]
    model_params["input_size"] = num_features

    model_params["output_size"] = config_data["ts_future"]


    epochs = config["training_params"]["epochs"]
    learning_rate = config["training_params"]["learning_rate"]
    optimizer_name = config["training_params"].get('optimizer')
    loss_function_name = config["training_params"].get('loss_function')

    print(f"epochs: {epochs}")
    print(f"learning_rate: {learning_rate}")
    print(f"optimizer_name: {optimizer_name}")
    print(f"loss_function_name: {loss_function_name}")
    # endregion

    # region ### INITIALIZE MODEL ###
    model = model_factory(model_type, **model_params)
    model.to(device)
    logging.info(model)
    # endregion

    # region ### DEFINE OPTIMIZER and LOSS FUNCTION ###
    # TODO shouldn't this reference to the dictionary ????
    if optimizer_name not in ['Adam', 'SGD']:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

    if loss_function_name not in ['MSELoss', 'CrossEntropyLoss']:
        raise ValueError(f"Invalid loss function name: {loss_function_name}")

    # Dictionary mapping for optimizers
    optimizers = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        # Add more optimizers as needed
        # TODO -> add more optimizers
    }
    # Dictionary mapping for loss functions
    loss_functions = {
        'MSELoss': torch.nn.MSELoss,
        'CrossEntropyLoss': torch.nn.CrossEntropyLoss,
        # Add more loss functions as needed
        # TODO -> add more loss functions
    }
    # Initialize optimizer and loss function
    optimizer = optimizers[optimizer_name](params=model.parameters(), lr=learning_rate)
    loss_fn = loss_functions[loss_function_name]()
    # endregion

    # region ### WRAP MODEL ###
    wrapper = wrapper_factory(model_type, model, optimizer, loss_fn)
    # endregion

    # region ### FIT / TRAIN MODEL ###
    model.train() # Set model to training mode
    training_loss, validation_loss = wrapper.fit(
        x_train, y_train, x_val, y_val, epochs, z_mean, z_std, selected_feature
    )
    # endregion

    # region ### REPORT VALIDATION LOSS TO RAY SESSION ###
    session.report({"val_loss": validation_loss[-1]})
    # endregion
# endregion

# Custom function to create shorter trial directory names
def custom_trial_dirname_creator(trial: Trial) -> str:
    # Use a simpler naming convention to reduce the length of the trial directory name
    return f"trial_{trial.trial_id}"

if __name__ == "__main__":
# TODO -> fix paths so command line arguments work

#    args = parse_args()
#    model_type = args.model_type
#    dataset = args.dataset
#    manual_hyperparams = args.manual_hyperparams
#    num_samples = args.num_samples
#    cpu_per_trial = args.cpu_per_trial
#    gpu_per_trial = args.gpu_per_trial

    model_type = 'RNN'
    dataset = 'sinusoidal'
    manual_hyperparams = False
    num_samples = 2
    cpu_per_trial = 10
    gpu_per_trial = 0.1


    # Load configuration for the dataset
    # TODO -> refactor so it is place maybe within OBJECTIVE
    config_data = load_config(get_path('config_dir', f'config_data_{dataset}.yaml'))
    ts_future = config_data['ts_future']

    # if manual_hyperparams:
    #     config = load_manual_hyperparameters()
    #     objective(config, config_data, model_type)
    # else:
    #     # Search space definition based on model type
    #     # TODO -> move to CONFIG FILE -> just one -> hyperparameter_config.json
    #     # TODO -> Then MANUAL one has the same structure but is loaded based on model + data
    #     def search_space():
    #         if model_type in ["RNN", "LSTM", "GRU"]:
    #             return {
    #                 "model_type": model_type,
    #                 "model_params": {
    #                     "output_size": ts_future,
    #                     "hidden_size": tune.randint(32, 64),
    #                     "num_layers": tune.randint(3, 4),
    #                     "dropout": tune.uniform(0.0, 0.5),
    #                 },
    #                 "training_params": {
    #                     "learning_rate": tune.loguniform(1e-3, 1e-2),
    #                     "epochs": tune.randint(50, 51),
    #                     "optimizer": tune.choice(['Adam', 'SGD']),
    #                     "loss_function": tune.choice(['MSELoss', 'CrossEntropyLoss']),
    #                 },
    #             }
    #         elif model_type == "Transformer":
    #             return {
    #                 "model_type": model_type,
    #                 "model_params": {
    #                     "output_size": ts_future,
    #                     "num_heads": tune.randint(2, 8),
    #                     "num_layers": tune.randint(1, 5),
    #                     "hidden_dim": tune.randint(64, 512),
    #                     "dropout": tune.uniform(0.0, 0.5),
    #                 },
    #                 "training_params": {
    #                     "learning_rate": tune.loguniform(1e-5, 1e-2),
    #                     "epochs": tune.randint(10, 50),
    #                     "optimizer": tune.choice(['Adam', 'SGD']),
    #                     "loss_function": tune.choice(['MSELoss', 'CrossEntropyLoss']),
    #                 },
    #             }


    # Initialize Ray
    ray.init(include_dashboard=false)

    # Run Ray Tune with search space defined
    # TODO -> is there a early STOP definition ??? like if after 10 epochs no improvement -> stop ??? !!!!
    analysis = tune.run(
        tune.with_parameters(objective, config_data=config_data, model_type=model_type),
        config=search_space(model_type),
        num_samples=num_samples,
        resources_per_trial={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
        trial_dirname_creator=custom_trial_dirname_creator,  # Custom trial directory naming
        resume=True,  # Resume from previous run
        max_concurrent_trials=1
    )

    # region ### SAVE BEST HYPERPARAMETERS ###
    # Get the best hyperparameters from tuning
    # TODO - add information about model / dataset / split / ts_future / ts_past --> everything which relates to training.... for loading and actual training
    best_config = analysis.get_best_config(metric="val_loss", mode="min")
    print("Best hyperparameter configuration:", best_config)
    best_config_file_path = get_path('config_dir', f'best_hyperparameters_{model_type}_{dataset}.json')
    # Save the best configuration to a JSON file
    with open(best_config_file_path, "w") as f:
        json.dump(best_config, f, indent=4)

    print(f"Best hyperparameters saved to {best_config_file_path}")
    # endregion


# Shutdown Ray
    ray.shutdown()