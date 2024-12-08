import argparse
import torch
import ray
import datetime

from ray import tune

# Callbacks
from manager.callback_manager import CallbackManager
from callbacks.logging_callback import LoggingCallback
from callbacks.plotting_callback import PlottingCallback
from callbacks.early_stopping_callback import EarlyStoppingCallback
from callbacks.tensorboard_callback import TensorBoardCallback

from callbacks.model_checkpoint_callback import ModelCheckpointCallback

# Strategies
from strategies.strategy_factory import StrategyFactory

# Manager
from manager.config_manager import ConfigManager
from manager.hyperparameter_manager import HyperparameterManager
from manager.file_manager import file_manager
from manager.path_manager import path_manager
from manager.experiment_manager import experiment_manager

from manager.data_manager import DataManager

from training_pipeline.training_pipeline_builder import TrainingPipelineBuilder

# Utilities
from utilities.tensorboard_utils import start_tensorboard
from utilities.experiment_utils import get_device
from utilities.experiment_utils import set_seed
from utilities.data_utils import load_and_preprocess_data, create_data_loaders

# Models and Pipeline
from models.model_factory import ModelFactory
from models.wrapper_factory import WrapperFactory


# Initialize logging
from utilities.logging_utils import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)
logger.info("Logger is set up")


# region ### PARSE ARGUMENTS ###
def parse_args():
    parser = argparse.ArgumentParser(description="Train different models with different datasets.")
    parser.add_argument("--config_path", type=str, default='config.yaml', help="Path to the configuration file.")

    logger.info(f'Parse Args: {parser.parse_args()}')
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
    # https://pytorch.org/docs/stable/nn.html#loss-functions
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
def objective(hyperparameter_config, model_type, data_manager):
    trial_seed = hyperparameter_config.get("trial_seed")
    set_seed(trial_seed)

    # region ### PREPARE DIRECTORY for TENSORBOARD ###
    # Generate a unique directory for each trial
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    tensorboard_trial_log_dir = tensorboard_log_dir / timestamp # tensorboard_log_dir from __main__
    logger.info(f"Tensorboard log directory for this trial: {tensorboard_trial_log_dir}")
    # endregion

    # region ### LOAD and PREPROCESS DATA ###
    data_manager.load_raw_data()
    data = data_manager.preprocess_data()

    # tensor_data = data['tensor_data']
    # array_data = data['array_data']
    metadata = data['metadata']

    num_features = metadata['num_features']
    z_mean, z_std, selected_feature = metadata['z_mean'], metadata['z_std'], metadata['selected_feature']
    # endregion

    # region ### SET VARIABLES from CONFIG ###
    model_params = hyperparameter_config["model_params"]
    model_params["input_size"] = num_features
    model_params["output_size"] = dataset_config["ts_future"]

    epochs = hyperparameter_config["training_params"]["epochs"]
    learning_rate = hyperparameter_config["training_params"]["learning_rate"]
    optimizer_name = hyperparameter_config["training_params"].get('optimizer')
    loss_function_name = hyperparameter_config["training_params"].get('loss_function')
    batch_size = hyperparameter_config["training_params"]["batch_size"]
    # endregion

    # region ### DATALOADER ###
    split_loader = data_manager.create_data_loaders(
        batch_size=batch_size, loaders_to_create=['train', 'val'], shuffle_train=True) # TODO if shuffle_train=True works
    train_loader, val_loader = split_loader['train_loader'], split_loader['val_loader']
    # endregion

    # region ### INITIALIZE MODEL and WRAPPER, OPTIMIZER, LOSS FUNCTION ###
    device = get_device()
    logger.info(f"Device: {device}")
    model = ModelFactory.create_model(model_type, **model_params).to(device)

    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
    loss_fn = get_loss_function(loss_function_name)

    wrapper = WrapperFactory.create_wrapper(model_type, model, optimizer, loss_fn, device)
    # endregion

    # region ### CALLBACKS ###
    logging_callback = LoggingCallback(log_every_n_epochs=1)
    plotting_callback = PlottingCallback(z_mean, z_std, selected_feature)
    early_stopping_callback = EarlyStoppingCallback(patience=10)

#    checkpoint_dir = path_manager.get_path_checkpoint(model_name=model_type, experiment_name=experiment_name)
#    model_checkpoint_callback = ModelCheckpointCallback(checkpoint_dir=checkpoint_dir, monitor='val_loss')

    hparams = hyperparameter_manager.get_hyperparameter_dict_for_tensorboard(hyperparameter_config) # Config chosen by RayTune

    # Create an instance of the ExtendedTensorBoardCallback
    tensorboard_callback = TensorBoardCallback(
        log_dir=tensorboard_trial_log_dir,
        hparams=hparams,
        model=model,
        example_input_tensor=torch.randn(ts_past, batch_size, num_features) # Create example input tensor so model graph can be logged to TensorBoard
    )

    callback_manager = CallbackManager(callbacks=[logging_callback, plotting_callback, early_stopping_callback, tensorboard_callback])
    # endregion

    # region ### Assemble the Training Pipeline ###
    builder = TrainingPipelineBuilder()
    training_pipeline = (builder
                         .set_model(model)
                         .set_wrapper(wrapper)
                         .set_data_loaders(train=train_loader, val=val_loader)
                         .set_epochs(epochs)
                         .set_callback_manager(callback_manager)
                         .build())
    # endregion

    # region ### Run the Training Pipeline ###
    training_pipeline.run()
    # endregion

# endregion



if __name__ == "__main__":
    args = parse_args()

    # Load the configuration file (->config.yaml)
    config = args.config_path
    config_manager = ConfigManager(config)
    data_manager = DataManager(config_manager)

    # Load parameters from the configuration file
    model_type = config_manager.get('model_type')
    dataset = config_manager.get('dataset')
    num_samples = config_manager.get('num_samples')
    cpu_per_trial = config_manager.get('cpu_per_trial')
    gpu_per_trial = config_manager.get('gpu_per_trial')
    max_concurrent_trials = config_manager.get('max_concurrent_trials')
    data_split = config_manager.get('train_val_test_split')
    trial_params = config_manager.get('trial_params')
    search_algorithm_name = config_manager.get("search_algorithm")
    scheduler_name = config_manager.get('scheduler')

    # Load configuration for the dataset (->config_data_{dataset}.yaml)
    dataset_config = config_manager.get_dataset_config()

    ts_future = dataset_config['ts_future']
    ts_past = dataset_config['ts_past']

    # Load hyperparameters for search space (->model_hyperparameter_config.json)
    # TODO - refactor that hyperparameter_config is also loaded through config_manager
    hyperparameter_config_path = path_manager.get_path_hyperparameter_config()
    hyperparameter_config = file_manager.load_json(hyperparameter_config_path)
    hyperparameter_manager = HyperparameterManager(hyperparameter_config)

    search_space_config = hyperparameter_manager.get_search_space(model_type)

    # Create search algorithm and scheduler using StrategyFactory
    search_algorithm = StrategyFactory.create_search_algorithm(search_algorithm_name, "val_loss", "min")
    scheduler = StrategyFactory.create_scheduler(scheduler_name, "val_loss", "min", trial_params)

    # Manage experiment: Check if it's a new experiment or should resume
    config_path = path_manager.get_path_config() / config
    experiment_name, resume = (
        experiment_manager.manage_experiment(model_type, dataset, ts_future, ts_past, config_path))

    # Set up the experiment paths
    experiment_path = path_manager.get_ray_results_dir() / experiment_name
    tensorboard_log_dir = path_manager.get_tensorboard_log_dir(model_type, experiment_name)

    logger.info(f"Experiment name: {experiment_name}")
    logger.info(f"Experiment path: {experiment_path}")
    logger.info(f"Resume: {resume}")
    logger.info(f"Tensorboard log directory: {tensorboard_log_dir}")

    # Start TensorBoard if needed (optional)
#    tensorboard_view_dir = path_manager.get_tensorboard_log_dir()
#    start_tensorboard(str(tensorboard_view_dir))

    # Initialize Ray
    ray.init(include_dashboard=False, local_mode=True) # CHANGE for local_mode=False for CLOUD

    # Run hyperparameter tuning with Ray Tune
    analysis = tune.run(
        tune.with_parameters(
            objective,
            model_type=model_type,
            data_manager=data_manager,
            ),
        config=search_space_config,
        num_samples=num_samples,
        resources_per_trial={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
        trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}",
        scheduler=scheduler,
        search_alg=search_algorithm,
        resume=True,
        max_concurrent_trials=max_concurrent_trials,
        name=experiment_name,
        storage_path=experiment_path
 #       log_to_file=True
    )


    # region ### SAVE BEST HYPERPARAMETERS ###
    best_config = analysis.get_best_config(metric="val_loss", mode="min")
    best_config_extended = {
        "model_type": model_type,
        "dataset": dataset,
        "data_split": data_split,
        "ts_future": ts_future,
        "ts_past": ts_past,
        "tuning_timestamp": datetime.datetime.now().isoformat(),
        "best_hyperparameters": best_config
    }

    experiment_manager.save_best_config(
        best_config_extended,
        model_name=model_type,
        dataset_name=dataset,
        ts_future=ts_future,
        ts_past=ts_past,
        split_id=data_split
    )

    # Now, to get the best trial, explicitly specify the metric and mode
    best_trial = analysis.get_best_trial(metric="val_loss", mode="min", scope="all")
    best_result = best_trial.last_result

    experiment_manager.generate_summary_report(best_config_extended, best_result)
    # endregion

    # Shutdown Ray
    ray.shutdown()