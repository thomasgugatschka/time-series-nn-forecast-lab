import argparse
import datetime
import torch

from src.callbacks.plotting_callback import PlottingCallback
from src.manager.callback_manager import CallbackManager
from src.manager.config_manager import ConfigManager
from src.manager.experiment_manager import ExperimentManager
from src.manager.data_manager import DataManager
from src.manager.plot_manager import PlotManager
from src.models.model_factory import ModelFactory
from src.models.wrapper_factory import WrapperFactory
from src.training_pipeline.training_pipeline_builder import TrainingPipelineBuilder
from src.utilities.experiment_utils import get_device, set_seed, get_optimizer, get_loss_function
from src.manager.file_manager import file_manager

from src.utilities.logging_utils import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# TODO -> USE the --num_models parameter in the .bat files !!!!
def parse_args():
    parser = argparse.ArgumentParser(description="Train models based on provided hyperparameters.")
    parser.add_argument("--config_path", type=str, default="config.yaml", help="Path to the configuration file.")
    parser.add_argument(
        "--num_models",
        type=restricted_int,
        default=5,
        help="Number of top hyperparameter configurations to use for training (1 to 10).",
    )
    return parser.parse_args()


def restricted_int(value):
    """Validate that the input is an integer within the range [1, 100]."""
    min_value = 1
    max_value = 10
    try:
        ivalue = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a valid integer.")

    if ivalue < min_value or ivalue > max_value:
        raise argparse.ArgumentTypeError(f"Value must be between {min_value} and {max_value}. You provided {value}.")

    return ivalue

def save_model(model, model_name, save_path, hyperparameter_config):
    """
    Save the trained model and its hyperparameters to files.

    Args:
        model (torch.nn.Module): Trained model instance.
        model_name (str): Name of the model.
        save_path (Path): Directory path to save the model and hyperparameters.
        hyperparameter_config (dict): Hyperparameter configuration used for training the model.
    """
    save_path.mkdir(parents=True, exist_ok=True)
    model_save_path = save_path / f"{model_name}.pt"
    hyperparams_save_path = save_path / f"{model_name}_hyperparameters.json"

    torch.save(model.state_dict(), model_save_path)
    file_manager.save_json(hyperparameter_config, hyperparams_save_path)

    logger.info(f"Model saved at {model_save_path}")
    logger.info(f"Hyperparameters saved at {hyperparams_save_path}")


# TODO #### TODO ##### TODO ####

# TODO -> SOMETHING IS NOT CORRECT with EPOCHS -> TRAINING vs. HYPERPARAMETER TUNING ?????
# TODO -> TRAINING MUCH MORE EPOCHS for SOME REASON ??????

# TODO -> probably need to USE   "training_iteration": xx isntead of "epoch"

# TODO -> !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

def train_model(hyperparameter_config, model_type, dataset_config, data_manager, plot_manager):
    """
    Train a single model based on the provided hyperparameter configuration.

    Args:
        hyperparameter_config (dict): Hyperparameter configuration for the model.
        model_type (str): Type of the model.
        dataset_config (dict): Dataset-specific configuration.
        data_manager (DataManager): Instance to manage data loading and preprocessing.
        plot_manager (PlotManager): Instance to manage plotting.
    Returns:
        torch.nn.Module: Trained model.
    """

    #TODO -> maybe train with different SEEDS ??? -> is it worth the check???
    set_seed(hyperparameter_config.get("config/trial_seed"))

    trial_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    # Load and preprocess data
    data_manager.load_raw_data()
    processed_data = data_manager.preprocess_data()
    num_features = processed_data["metadata"]["num_features"]

    # Set model parameters
    model_params = {
        "input_size": num_features,
        "output_size": dataset_config["ts_future"],
        "seq_len": dataset_config["ts_past"],
        **{
            k.split("/")[-1]: v for k, v in hyperparameter_config.items() if k.startswith("config/model_params")
        }
    }

    # Training parameters
    training_params = {
        k.split("/")[-1]: v for k, v in hyperparameter_config.items() if k.startswith("config/training_params")
    }
    epochs = training_params["epochs"]
    batch_size = training_params["batch_size"]

    # Create data loaders
    loaders = data_manager.create_data_loaders(batch_size=batch_size, loaders_to_create=["train", "val"], shuffle_train=True)
    train_loader, val_loader = loaders["train_loader"], loaders["val_loader"]

    # Initialize model, optimizer, and loss function

    device = get_device()
    model = ModelFactory.create_model(model_type, **model_params).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=training_params["learning_rate"])
    #loss_fn = torch.nn.MSELoss()
    optimizer = get_optimizer(training_params["optimizer"], model.parameters(), training_params["learning_rate"])
    loss_fn = get_loss_function(training_params["loss_function"])

    wrapper = WrapperFactory.create_wrapper(model_type, model, optimizer, loss_fn, device)

    # Callbacks
    plotting_callback = PlottingCallback(plot_manager, trial_name)
    callback_manager = CallbackManager(callbacks=[plotting_callback])

    # Assemble training pipeline
    builder = TrainingPipelineBuilder()
    training_pipeline = (
        builder
        .set_model(model)
        .set_wrapper(wrapper)
        .set_data_loaders(train=train_loader, val=val_loader)
        .set_epochs(epochs)
        .set_callback_manager(callback_manager)
        .build()
    )

    # Train the model
    training_pipeline.run()

    return model


def main():
    args = parse_args()

    # Load configurations
    config_manager = ConfigManager(args.config_path)
    data_manager = DataManager(config_manager)
    experiment_manager = ExperimentManager(config_manager)
    experiment_name, experiment_path, _ = experiment_manager.manage_experiment()
    plot_manager = PlotManager(config_manager, data_manager, phase="training")

    dataset_config = config_manager.get_dataset_config()
    main_config = config_manager.get_main_config()
    model_type = main_config["model_type"]

    # Load top hyperparameters from experiment folder

    hyperparameter_config_path = experiment_path / "top_hyperparameters.json"
    top_hyperparameters = file_manager.load_json(hyperparameter_config_path)

    # Select top configurations
    top_hyperparameters = top_hyperparameters[: args.num_models]

    # Model save path
    model_save_path = experiment_path / "trained_models"

    # Train and save models
    for i, hyperparameter_config in enumerate(top_hyperparameters):
        logger.info(f"Training model {i + 1}/{args.num_models} with configuration: {hyperparameter_config}")
        model = train_model(hyperparameter_config, model_type, dataset_config, data_manager, plot_manager)
        val_loss = hyperparameter_config.get("val_loss", "unknown")
        model_name = f"{model_type}_{i + 1}_val_loss={val_loss:.4f}".replace("/", "-")
        save_model(model, model_name, model_save_path, hyperparameter_config)


if __name__ == "__main__":
    main()
