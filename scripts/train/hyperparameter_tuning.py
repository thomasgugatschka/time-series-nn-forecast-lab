import argparse
import torch
import logging
import ray
import datetime


from ray import tune
from ray.tune.experiment.trial import Trial

# Callbacks
from manager.callback_manager import CallbackManager
from callbacks.logging_callback import LoggingCallback
from callbacks.plotting_callback import PlottingCallback
from callbacks.early_stopping_callback import EarlyStoppingCallback

# Manager
from manager.config_manager import ConfigManager
from manager.hyperparameter_manager import HyperparameterManager
from manager.file_manager import file_manager
from manager.path_manager import path_manager
from manager.result_manager import result_manager

from configs.logging_config import setup_logging

# Utilities
from utilities.training_utils import get_device
from utilities.training_utils import set_seed
from utilities.data_utils import load_and_preprocess_data, create_data_loaders

from utilities.scheduler_factory import get_scheduler

# Models and Pipeline
from models.model_factory import model_factory
from models.wrapper_factory import wrapper_factory
from strategies.search_algorithm_strategy import SearchAlgorithmStrategy
from training_pipeline.hyperparameter_tuning_pipeline import TrainingPipeline


# Initialize logging
setup_logging()


# region ### PARSE ARGUMENTS ###
def parse_args():
    parser = argparse.ArgumentParser(description="Train different models with different datasets.")
    parser.add_argument("--config_path", type=str, default='config.yaml', help="Path to the configuration file.")

    logging.info(f'Parse Args: {parser.parse_args()}')
    return parser.parse_args()
# endregion

# region ### UTILITIES ###
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

# endregion

# region ### TRAINING FUNCTION ##
def objective(config, dataset_config, model_type, default_data_split):
    trial_seed = config.get("trial_seed")
    set_seed(trial_seed)


    # region ### LOAD and PREPROCESS DATA ###
    data = load_and_preprocess_data(dataset_config, dataset, default_data_split)

    x_train, x_val, x_test = data["x_train"], data["x_val"], data["x_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    num_features = data["num_features"]
    z_mean, z_std, selected_feature = data["z_mean"], data["z_std"], data["selected_feature"]
    # endregion

    # region ### SET VARIABLES from CONFIG ###
    model_params = config["model_params"]
    model_params["input_size"] = num_features
    model_params["output_size"] = dataset_config["ts_future"]

    epochs = config["training_params"]["epochs"]
    learning_rate = config["training_params"]["learning_rate"]
    optimizer_name = config["training_params"].get('optimizer')
    loss_function_name = config["training_params"].get('loss_function')
    batch_size = config["training_params"]["batch_size"]
    # endregion

    # region ### DATALOADER ###
    train_loader, val_loader, test_loader = create_data_loaders(
        x_train, y_train, x_val, y_val, x_test, y_test, batch_size
    )
    # endregion

    # region ### INITIALIZE MODEL, OPTIMIZER, LOSS FUNCTION ###
    device = get_device()
    logging.info(f"Device: {device}")
    model = model_factory(model_type, **model_params).to(device)
    #logging.info(model)

    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
    loss_fn = get_loss_function(loss_function_name)
    # endregion

    # region ### CALLBACKS ###
    # Set up callbacks using the callback manager
    logging_callback = LoggingCallback(log_every_n_epochs=1)
    plotting_callback = PlottingCallback(z_mean, z_std, selected_feature)
    early_stopping_callback = EarlyStoppingCallback(patience=10)

    checkpoint_dir = path_manager.get_path_checkpoint(model_name=model_type) # TODO maybe need to add dataset? and also split?
    #model_checkpoint_callback = ModelCheckpointCallback(checkpoint_dir=checkpoint_dir, monitor='val_loss')

    callback_manager = CallbackManager(callbacks=[logging_callback, plotting_callback, early_stopping_callback])
    # endregion

    # region ### INITIALIZE MODEL WRAPPER ###
    wrapper = wrapper_factory(model_type, model, optimizer, loss_fn, device)
    # endregion

    # region ### Assemble the pipeline config ###
    pipeline_config = {
        "model": model,
        "data_loaders": {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        },
        "epochs": epochs,
        "callback_manager": callback_manager
    }
    # endregion

    # region ### Create the TrainingPipeline ###
    training_pipeline = TrainingPipeline(pipeline_config, wrapper)
    training_pipeline.run()
    # endregion

# endregion



if __name__ == "__main__":
    args = parse_args()

    # Load the configuration file (->config.yaml)
    config_manager = ConfigManager(args.config_path)

    model_type = config_manager.get('model_type')
    dataset = config_manager.get('dataset')
    num_samples = config_manager.get('num_samples')
    cpu_per_trial = config_manager.get('cpu_per_trial')
    gpu_per_trial = config_manager.get('gpu_per_trial')
    max_concurrent_trials = config_manager.get('max_concurrent_trials')
    default_data_split = config_manager.get('train_val_test_split')
    scheduler_name = config_manager.get('scheduler')
    trial_params = config_manager.get('trial_params')
    search_algorithm_name = config_manager.get("search_algorithm")

    # Load configuration for the dataset (-> config_data_{dataset}.yaml)
    dataset_config_path = path_manager.get_path_dataset_config(dataset_name=dataset)
    dataset_config = file_manager.load_yaml(dataset_config_path)

    # Load hyperparameters (->model_hyperparameter_config.json)
    hyperparameter_config_path = path_manager.get_path_hyperparameter_config()
    hyperparameter_config = file_manager.load_json(hyperparameter_config_path)

    hyperparameter_manager = HyperparameterManager(hyperparameter_config)





    # TODO - refactor don't like it here, tooo random
    ts_future = dataset_config['ts_future']




    scheduler = get_scheduler(scheduler_name, trial_params)

    # Run Ray Tune with search space defined


    # Initialize Ray
    ray.init(include_dashboard=False)

# TODO REFACTOR SEARCH ALGORITHM... maybe not strategy pattern?

    # Search algorithm setup using the SearchAlgorithmStrategy
    search_algorithm_strategy = SearchAlgorithmStrategy(
        algorithm_name=search_algorithm_name,
        metric="val_loss",
        mode="min"
    )
    search_algorithm = search_algorithm_strategy.get_search_algorithm()

    search_space_config = hyperparameter_manager.get_search_space(model_type)

    # Run hyperparameter tuning with Ray Tune
    analysis = tune.run(
        tune.with_parameters(
            objective,
            dataset_config=dataset_config,
            model_type=model_type,
            default_data_split=default_data_split),
        config=search_space_config,
        num_samples=num_samples,
        resources_per_trial={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
        trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
        scheduler=scheduler,
        search_alg=search_algorithm,
        #resume=True,
        resume=False,
        max_concurrent_trials=max_concurrent_trials
    )




    # region ### SAVE BEST HYPERPARAMETERS ###
    best_config = analysis.get_best_config(metric="val_loss", mode="min")
    best_config_extended = {
        "model_type": model_type,
        "dataset": dataset,
        "train_val_test_split": default_data_split,
        "ts_future": dataset_config.get('ts_future'),
        "ts_past": dataset_config.get('ts_past'),
        "tuning_timestamp": datetime.datetime.now().isoformat(),
        "best_hyperparameters": best_config
    }

    result_manager.save_best_config(
        best_config_extended,
        model_name=model_type,
        dataset_name=dataset,
        ts_future=dataset_config.get('ts_future'),
        ts_past=dataset_config.get('ts_past'),
        split_id="default"
    )

    print(f"Best hyperparameters: {best_config_extended}")
    # endregion

    # region ### LOG TO TENSORBOARD (OPTIONAL) ###
    if config_manager.get('log_to_tensorboard', default=False):
        best_trial = analysis.get_best_trial(metric="val_loss", mode="min")
        best_metric_value = best_trial.last_result["val_loss"] if best_trial else None

        # Use the ResultManager to log hyperparameters and best metric to TensorBoard
        if best_metric_value is not None:
            result_manager.log_to_tensorboard(model_type, best_config_extended, best_metric_value)
    # endregion


    # Shutdown Ray
    ray.shutdown()