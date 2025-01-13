import argparse
import torch
import ray
import datetime

from ray import tune

# Callbacks
from src.manager.callback_manager import CallbackManager
from src.callbacks.logging_callback import LoggingCallback
from src.callbacks.plotting_callback import PlottingCallback
from src.callbacks.early_stopping_callback import EarlyStoppingCallback
from src.callbacks.tensorboard_callback import TensorBoardCallback

# Strategies
from src.strategies.strategy_factory import StrategyFactory

# Manager
from src.manager.config_manager import ConfigManager
from src.manager.hyperparameter_manager import HyperparameterManager
from src.manager.experiment_manager import ExperimentManager
from src.manager.plot_manager import PlotManager
from src.manager.data_manager import DataManager
from src.manager.env_manager import EnvManager
from src.manager.file_manager import file_manager
from src.manager.path_manager import path_manager

# Utilities
from src.utilities.experiment_utils import set_seed
from src.models.model_utils import get_device, get_optimizer, get_loss_function


# Models and Pipeline
from src.models.model_factory import ModelFactory
from src.models.wrapper_factory import WrapperFactory
from src.training_pipeline.training_pipeline_builder import TrainingPipelineBuilder

# Initialize logging
from src.utilities.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger(__name__)


# region ### PARSE ARGUMENTS ###
def parse_args():
    """
    Parse command-line arguments for hyperparameter tuning.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train different models with different datasets.")
    parser.add_argument("--config_path", type=str, default='config.yaml', help="Path to the configuration file.")
    logger.info(f'Parse Args: {parser.parse_args()}')
    return parser.parse_args()
# endregion

# region ### TRAINING FUNCTION ##
def objective(hyperparameter_config, model_type, dataset_config, data_manager, plot_manager):
    """
    Objective function for hyperparameter tuning.

    Args:
        hyperparameter_config (dict): Configuration for the trial.
        model_type (str): Type of the model being tuned.
        dataset_config (dict): Dataset-specific configuration.
        data_manager (DataManager): Manager for data loading and preprocessing.
        plot_manager (PlotManager): Manager for creating plots during the trial.
    """

    set_seed(hyperparameter_config.get("trial_seed"))
    device = get_device()

    # Workaround... could not pass trail name to the objective function
    trial_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    # region ### LOAD and PREPROCESS DATA ###
    data_manager.load_raw_data()
    data = data_manager.preprocess_data()
    num_features = data['metadata']['num_features']
    # endregion

    # region ### SET VARIABLES from CONFIG ###
    # TODO -> Change naming of model_params -> confusing if input_size is num_features and output_size is ts_future
    ts_future = dataset_config["ts_future"]
    ts_past = dataset_config["ts_past"]

    model_params = hyperparameter_config["model_params"]
    model_params["input_size"] = num_features
    model_params["output_size"] = ts_future
    model_params["seq_len"] = ts_past

    epochs = hyperparameter_config["training_params"]["epochs"]
    learning_rate = hyperparameter_config["training_params"]["learning_rate"]
    optimizer_name = hyperparameter_config["training_params"].get('optimizer')
    loss_function_name = hyperparameter_config["training_params"].get('loss_function')
    batch_size = hyperparameter_config["training_params"]["batch_size"]
    # endregion

    # region ### DATALOADER ###
    split_loader = data_manager.create_data_loaders(
        batch_size=batch_size, loaders_to_create=['train', 'val'], shuffle_train=False) # IMPROVEMENT: / OPTION: shuffle_train=True -> Check if it makes sense for time series
    train_loader, val_loader = split_loader['train_loader'], split_loader['val_loader']
    # endregion

    # region ### INITIALIZE MODEL and WRAPPER, OPTIMIZER, LOSS FUNCTION ###
    model = ModelFactory.create_model(model_type, **model_params).to(device)
    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate)
    loss_fn = get_loss_function(loss_function_name)
    wrapper = WrapperFactory.create_wrapper(model_type, model, optimizer, loss_fn, device)
    # endregion

    # region ### CALLBACKS ###
    logging_callback = LoggingCallback(log_every_n_epochs=1)
    plotting_callback = PlottingCallback(plot_manager, trial_name)
    early_stopping_callback = EarlyStoppingCallback(patience=10)
    tensorboard_callback = TensorBoardCallback(
        #log_dir=experiment_path / "tensorboard" / trial_name,
        log_dir=path_manager.get_path('experiment_tensorboard') / trial_name,
        hparams=hyperparameter_config,
        model=model,
        example_input_tensor=torch.randn(batch_size, ts_past, num_features)
    )
    #    checkpoint_dir = path_manager.get_path_checkpoint(model_name=model_type, experiment_name=experiment_name)
    #    model_checkpoint_callback = ModelCheckpointCallback(checkpoint_dir=checkpoint_dir, monitor='val_loss')

    callback_manager = CallbackManager(
        callbacks=[logging_callback, plotting_callback, early_stopping_callback, tensorboard_callback])
    # endregion

    # region ### ASSEMBLE and RUN the TRAINING PIPELINE ###
    builder = TrainingPipelineBuilder()
    training_pipeline = (builder
                         .set_model(model)
                         .set_wrapper(wrapper)
                         .set_data_loaders(train=train_loader, val=val_loader)
                         .set_epochs(epochs)
                         .set_callback_manager(callback_manager)
                         .build())

    training_pipeline.run()
    # endregion
# endregion

# region ### MAIN FUNCTION ###
def main():
    args = parse_args()

    config_manager = ConfigManager(args.config_path)
    data_manager = DataManager(config_manager)
    env_manager = EnvManager()
    experiment_manager = ExperimentManager(config_manager)
    experiment_name, experiment_path, resume = experiment_manager.manage_experiment(phase='tuning')
    plot_manager = PlotManager(config_manager, data_manager, phase='tuning')

    main_config = config_manager.get_main_config()
    dataset_config = config_manager.get_dataset_config()
    hyperparameter_config = file_manager.load_json(path_manager.get_path("config_dir") / "hyperparameter_search_space.json")
    hyperparameter_manager = HyperparameterManager(hyperparameter_config)

    # Experiment Paths
    logger.info(f"Experiment name: {experiment_name} || Resume: {resume}")
    logger.info(f"Experiment path: {experiment_path}")


    # Load parameters from the configuration file
    model_type = main_config.get('model_type')
    num_samples = main_config.get('num_samples')
    cpu_per_trial = main_config.get('cpu_per_trial')
    gpu_per_trial = main_config.get('gpu_per_trial')
    max_concurrent_trials = main_config.get('max_concurrent_trials')
    trial_params = main_config.get('trial_params')
    search_algorithm_name = main_config.get("search_algorithm")
    scheduler_name = main_config.get('scheduler')

    # Get search space configuration for the model
    search_space_config = hyperparameter_manager.get_search_space(model_type)

    # Create search algorithm and scheduler using StrategyFactory
    search_algorithm = StrategyFactory.create_search_algorithm(search_algorithm_name, "val_loss", "min")
    scheduler = StrategyFactory.create_scheduler(scheduler_name, "val_loss", "min", trial_params)

    local_mode = env_manager.get_env_variable("LOCAL_MODE", default="True", as_bool=True)

    # Initialize Ray
    ray.init(include_dashboard=False, local_mode=local_mode)

    try:
        # Run hyperparameter tuning with Ray Tune
        analysis = tune.run(
            tune.with_parameters(
                objective,
                model_type=model_type,
                dataset_config=dataset_config,
                data_manager=data_manager,
                plot_manager=plot_manager
                ),
            config=search_space_config,
            num_samples=num_samples,
            resources_per_trial={"cpu": cpu_per_trial, "gpu": gpu_per_trial},
            trial_dirname_creator=lambda trial: f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}_{trial.trial_id}",
            scheduler=scheduler,
            search_alg=search_algorithm,
            resume=resume,
            max_concurrent_trials=max_concurrent_trials,
            name="ray_results",
            storage_path=experiment_path
        )

        experiment_manager.save_analysis_results(analysis)

    except Exception as e:
        logger.error(f"Error during hyperparameter tuning or saving results: {e}")
        raise e
    finally:
        try:
            experiment_manager.finalize_experiment()
        except Exception as finalize_error:
            logger.error(f"Error during experiment finalization: {finalize_error}", exc_info=True)

        try:
            ray.shutdown()
        except Exception as shutdown_error:
            logger.error(f"Error during Ray shutdown: {shutdown_error}", exc_info=True)


# endregion

if __name__ == "__main__":
    main()