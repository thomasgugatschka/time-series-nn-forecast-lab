import argparse

import torch
import time
import logging
import ray
import json

from ray import tune
from ray.tune.experiment.trial import Trial
from ray.tune.schedulers import ASHAScheduler
from ray.air import session
from sympy import false


from callbacks.logging_callback import LoggingCallback
from callbacks.plotting_callback import PlottingCallback
from configs.logging_config import setup_logging
from configs.paths import get_path

from utilities.device_utils import get_device
from utilities.seed_utils import set_seed
from utilities.save_utils import save_parameters_and_results
from utilities.data_utils import preprocess_data, split_and_convert_data, z_score_denormalize, load_config
from utilities.plot_utils import plot_training_progress, plot_prediction, plot_prediction_with_context, create_gif_from_plots

from models.model_factory import model_factory
from models.wrapper_factory import wrapper_factory


setup_logging()

# Load the configuration file
# TODO -> refactor naming.... so all config etc. have same same naming convention
# TODO REFACOTR -> so Load Config or so can be used... myaybe new util for files / file loading ?
with open(get_path('config_dir', 'hyperparameter_config.json'), "r") as f:
    config_hyperparameter = json.load(f)




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
        "trial_seed": tune.randint(0, 10000),  # Assign a different seed for each trial
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
    parser.add_argument("--max_concurrent_trials", type=int, default=1, help="Number of concurrent trials")
    return parser.parse_args()
# endregion


# region ### TRAINING FUNCTION ##
def objective(config, config_data, model_type):
    trial_seed = config.get("trial_seed")
    set_seed(trial_seed)

    device = get_device()
    logging.info(f"Device: {device}")

    # region ### LOAD and PREPROCESS DATA ###
    df, x_arr, y_arr, z_mean, z_std, selected_feature, num_features = preprocess_data(config_data, dataset)
    # endregion

    # region ### SPLIT and CONVERT DATA ###
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
    # endregion

    # region ### INITIALIZE MODEL ###
    model = model_factory(model_type, **model_params)
    model.to(device)
    logging.info(model)
    # endregion

    # region ### DEFINE OPTIMIZER and LOSS FUNCTION ###
    # Dictionary mapping for optimizers
    optimizers = {
        'Adam': torch.optim.Adam,
        'SGD': torch.optim.SGD,
        'AdamW': torch.optim.AdamW,
        'RMSprop': torch.optim.RMSprop,
        'Adagrad': torch.optim.Adagrad,
        'AdaDelta': torch.optim.Adadelta,
        'NAdam': torch.optim.NAdam,
        'Adamax': torch.optim.Adamax,
        'ASGD': torch.optim.ASGD,
    }
    # Dictionary mapping for loss functions
    loss_functions = {
        'MSELoss': torch.nn.MSELoss,
        'MAELoss': torch.nn.L1Loss,
        'HuberLoss': torch.nn.SmoothL1Loss,
        'NLLLoss': torch.nn.NLLLoss,
    }

    # Reference to the dictionary instead of hardcoding
    if optimizer_name not in optimizers:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")

    if loss_function_name not in loss_functions:
        raise ValueError(f"Invalid loss function name: {loss_function_name}")

    # Initialize optimizer and loss function
    optimizer = optimizers[optimizer_name](params=model.parameters(), lr=learning_rate)
    loss_fn = loss_functions[loss_function_name]()
    # endregion

    # region ### CALLBACKS ###
    # Create callback instance #TODO for TRAIN SCRIPT RELEVANT
    logging_callback = LoggingCallback(log_every_n_epochs=1)
    plotting_callback = PlottingCallback(z_mean, z_std, selected_feature)
    # endregion

    # region ### WRAP MODEL ###
    #wrapper = wrapper_factory(model_type, model, optimizer, loss_fn, callbacks=[logging_callback, plotting_callback]) #TODO for TRAIN SCRIPT RELEVANT
    wrapper = wrapper_factory(model_type, model, optimizer, loss_fn)
    # endregion

    # region ### FIT / TRAIN MODEL ###
    model.train() # Set model to training mode
    training_loss, validation_loss = wrapper.fit(x_train, y_train, x_val, y_val, epochs)
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
# TODO - fix paths so command line arguments work

#    args = parse_args()
#    model_type = args.model_type
#    dataset = args.dataset
#    num_samples = args.num_samples
#    cpu_per_trial = args.cpu_per_trial
#    gpu_per_trial = args.gpu_per_trial
#    max_concurrent_trials = args.max_concurrent_trials

    model_type = 'RNN'
    dataset = 'sinusoidal'
    num_samples = 2
    cpu_per_trial = 10
    gpu_per_trial = 0.1
    max_concurrent_trials = 1  # all "trial" resources are multiplied by the number of trials


    # Load configuration for the dataset
    # TODO -> refactor so it is place maybe within OBJECTIVE
    config_data = load_config(get_path('config_dir', f'config_data_{dataset}.yaml'))
    ts_future = config_data['ts_future']


    # Initialize Ray
    ray.init(include_dashboard=false)

    # Run Ray Tune with search space defined

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=50,  # max epochs
        grace_period=5,  # minimum number of epochs before stopping poorly performing trials
        reduction_factor=2
    )

    analysis = tune.run(
        tune.with_parameters(objective, config_data=config_data, model_type=model_type),
        config=search_space(model_type),
        num_samples=num_samples,
        resources_per_trial={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
        trial_dirname_creator=custom_trial_dirname_creator,  # Custom trial directory naming
        scheduler=scheduler,
        resume=True,  # Resume from previous run
        max_concurrent_trials=max_concurrent_trials
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