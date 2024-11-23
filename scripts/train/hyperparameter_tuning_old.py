import argparse
import torch
import logging
import ray
import json

from ray import tune
from ray.air import session
from ray.tune.experiment.trial import Trial
from ray.tune.schedulers import ASHAScheduler

from manager.callback_manager import CallbackManager
from callbacks.logging_callback import LoggingCallback
from callbacks.plotting_callback import PlottingCallback
from callbacks.early_stopping_callback import EarlyStoppingCallback

from manager.config_manager import ConfigManager
from manager.hyperparameter_manager import HyperparameterManager

from configs.logging_config import setup_logging
from manager.path_manager import get_path

from utilities.training_utils import get_device
from utilities.training_utils import set_seed
from utilities.data_utils import load_config, load_and_preprocess_data, create_data_loaders

from models.model_factory import model_factory
from models.wrapper_factory import wrapper_factory


setup_logging()


# region ### PARSE ARGUMENTS ###
# TODO: Updated parse_args to optionally allow override of default config path
def parse_args():
    parser = argparse.ArgumentParser(description="Train different models with different datasets.")
    parser.add_argument("--config_path", type=str, default='configs/config.yaml', help="Path to the configuration file.")
    return parser.parse_args()

#def parse_args():
#    parser = argparse.ArgumentParser(description="Train different models with different datasets.")
#    parser.add_argument("--model_type", type=str, required=True, help="Model type to train (RNN, LSTM, GRU, Transformer, CNN)")
#    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (e.g., sinusoidal, weather_station)")
#    parser.add_argument("--num_samples", type=int, default=10, help="Number of trials for hyperparameter tuning")
#    parser.add_argument("--cpu_per_trial", type=int, default=2, help="Number of CPUs per trial")
#    parser.add_argument("--gpu_per_trial", type=float, default=0.5, help="Number of GPUs per trial")
#    parser.add_argument("--max_concurrent_trials", type=int, default=1, help="Number of concurrent trials")
#    return parser.parse_args()
# endregion

def get_optimizer(optimizer_name, params, lr):
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
    if optimizer_name not in optimizers:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}")
    return optimizers[optimizer_name](params=params, lr=lr)

def get_loss_function(loss_function_name):
    loss_functions = {
        'MSELoss': torch.nn.MSELoss,
        'MAELoss': torch.nn.L1Loss,
        'HuberLoss': torch.nn.SmoothL1Loss,
        'NLLLoss': torch.nn.NLLLoss,
    }
    if loss_function_name not in loss_functions:
        raise ValueError(f"Invalid loss function name: {loss_function_name}")
    return loss_functions[loss_function_name]()


# region ### TRAINING FUNCTION ##
def objective(config, config_data, model_type, default_split):
    trial_seed = config.get("trial_seed")
    set_seed(trial_seed)

    device = get_device()
    logging.info(f"Device: {device}")


    # region ### LOAD and PREPROCESS DATA ###
    data = load_and_preprocess_data(config_data, dataset, default_split)

    x_train, x_val, x_test = data["x_train"], data["x_val"], data["x_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    num_features = data["num_features"]
    z_mean, z_std, selected_feature = data["z_mean"], data["z_std"], data["selected_feature"]
    # endregion


    # region ### SET VARIABLES from CONFIG ###
    model_params = config["model_params"]
    model_params["input_size"] = num_features
    model_params["output_size"] = config_data["ts_future"]

    epochs = config["training_params"]["epochs"]
    learning_rate = config["training_params"]["learning_rate"]
    optimizer_name = config["training_params"].get('optimizer')
    loss_function_name = config["training_params"].get('loss_function')

    # Get batch size from config
    batch_size = config["training_params"]["batch_size"]

    # endregion

    # region ### DATALOADER ###
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        x_train, y_train, x_val, y_val, x_test, y_test, batch_size
    )
    # endregion



    # region ### INITIALIZE MODEL ###
    model = model_factory(model_type, **model_params)
    model.to(device)
    logging.info(model)
    # endregion

    # region ### DEFINE OPTIMIZER and LOSS FUNCTION ###
    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
    loss_fn = get_loss_function(loss_function_name)
    # endregion





    # region ### CALLBACKS ###
    # Set up callbacks using the callback manager
    logging_callback = LoggingCallback(log_every_n_epochs=1)
    plotting_callback = PlottingCallback(z_mean, z_std, selected_feature)
    early_stopping_callback = EarlyStoppingCallback(patience=10)

    checkpoint_dir = get_path('checkpoint_dir', model_type) # TODO maybe need to add dataset? and also split?
    #model_checkpoint_callback = ModelCheckpointCallback(checkpoint_dir=checkpoint_dir, monitor='val_loss')

    callback_manager = CallbackManager(callbacks=[logging_callback, plotting_callback, early_stopping_callback])

    # region ### WRAP MODEL ###
    wrapper = wrapper_factory(model_type, model, optimizer, loss_fn)
    # endregion

    # region ### FIT / TRAIN MODEL ###
    # Training Loop with DataLoader Integration and Callbacks
    for epoch in range(epochs):
        logging.info(f'Start Training Epoch {epoch}')

        # Initialize variables to track epoch losses
        train_loss_total = 0
        num_train_batches = 0

        # Training Loop for each mini-batch
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            train_loss = wrapper.train_epoch(x_batch, y_batch)
            train_loss_total += train_loss
            num_train_batches += 1

        avg_train_loss = train_loss_total / num_train_batches
        logging.info(f'Epoch {epoch}: avg_train_loss = {avg_train_loss:.4f}')

        # Validation Loop
        val_loss_total = 0
        num_val_batches = 0
        for x_val_batch, y_val_batch in val_loader:
            x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
            val_loss = wrapper.validate(x_val_batch, y_val_batch)
            val_loss_total += val_loss
            num_val_batches += 1

        avg_val_loss = val_loss_total / num_val_batches
        # TODO - move to callback manager
        logging.info(f'Epoch {epoch}: avg_val_loss = {avg_val_loss:.4f}')

        # Use the callback manager to trigger on_epoch_end
        callback_manager.on_epoch_end(epoch, wrapper, avg_val_loss, x_val=x_val, y_val=y_val)

        # Check if early stopping is triggered
        if early_stopping_callback.should_stop():
            logging.info(f"Training stopped early at epoch {epoch + 1}")
            break

        # Report the validation loss to Ray Tune
        session.report({"val_loss": avg_val_loss})
    # endregion

# endregion

# Custom function to create shorter trial directory names
def custom_trial_dirname_creator(trial: Trial) -> str:
    # Use a simpler naming convention to reduce the length of the trial directory name
    return f"trial_{trial.trial_id}"




if __name__ == "__main__":

    args = parse_args()
    config_manager = ConfigManager(args.config_path)

    # Load hyperparameters
    with open(get_path('config_dir', 'hyperparameter_config.json'), "r") as f:
        config_hyperparameter = json.load(f)

    hyperparameter_manager = HyperparameterManager(config_hyperparameter)

    model_type = config_manager.get('model_type')
    dataset = config_manager.get('dataset')
    num_samples = config_manager.get('num_samples')
    cpu_per_trial = config_manager.get('cpu_per_trial')
    gpu_per_trial = config_manager.get('gpu_per_trial')
    max_concurrent_trials = config_manager.get('max_concurrent_trials')

    # Load the default split ratios from the experiment config
    default_split = config_manager.get('train_val_test_split')

    # Load configuration for the dataset
    config_data = load_config(get_path('config_dir', f'config_data_{dataset}.yaml'))



    ts_future = config_data['ts_future']



    # Run Ray Tune with search space defined


    # Initialize Ray
    ray.init(include_dashboard=False)

    # Run Ray Tune with search space defined
    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=config_manager['trial_params']['max_epochs'],
        grace_period=config_manager['trial_params']['grace_period'],
        reduction_factor=config_manager['trial_params']['reduction_factor']
    )

    search_space_config = hyperparameter_manager.get_search_space(model_type)

    analysis = tune.run(
        tune.with_parameters(objective, config_data=config_data, model_type=model_type, default_split=default_split),
        config=search_space_config,
        num_samples=num_samples,
        resources_per_trial={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
        trial_dirname_creator=custom_trial_dirname_creator,
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