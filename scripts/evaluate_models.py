import argparse
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from manager.config_manager import ConfigManager
from manager.data_manager import DataManager
from manager.file_manager import file_manager
from manager.experiment_manager import ExperimentManager
from models.model_factory import ModelFactory
from models.wrapper_factory import WrapperFactory
from utilities.experiment_utils import get_device
from utilities.metrics_utils import MetricsUtility

# Initialize logging
from utilities.logging_utils import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

def load_model(model_path, model_type, model_params, device):
    """
    Load a trained model from a file.

    Args:
        model_path (Path): Path to the model file.
        model_type (str): Type of the model.
        model_params (dict): Model parameters.
        device (torch.device): Device to load the model onto.

    Returns:
        torch.nn.Module: The loaded model.
    """
    model = ModelFactory.create_model(model_type, **model_params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def plot_predictions(ts_past, ts_future, x_context, y_actual, y_predicted, output_path, plot_title):
    """
    Plot and save the actual vs. predicted values with context.

    Args:
        ts_past (int): Number of past time steps.
        ts_future (int): Number of future time steps.
        x_context (numpy.ndarray): Context data.
        y_actual (numpy.ndarray): Actual values.
        y_predicted (numpy.ndarray): Predicted values.
        output_path (Path): Path to save the plot.
        plot_title (str): Title of the plot including model and hyperparameter details.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(-ts_past, 0), x_context, label="Context", color="blue")
    plt.plot(range(0, ts_future), y_actual, label="Actual", color="green")
    plt.plot(range(0, ts_future), y_predicted, label="Predicted", color="red")
    plt.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
    plt.legend()
    plt.title(plot_title)
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.savefig(output_path)
    plt.close()

def evaluate_model(model, wrapper, test_data, ts_past, ts_future, output_dir, hyperparameters, model_name):
    """
    Evaluate a single model on the test dataset.

    Args:
        model (torch.nn.Module): The trained model.
        wrapper: Wrapper for handling model operations.
        test_data (dict): Test data containing input and target tensors.
        ts_past (int): Number of past time steps.
        ts_future (int): Number of future time steps.
        output_dir (Path): Directory to save evaluation results.
        hyperparameters (dict): Hyperparameters used to train the model.
        model_name (str): Name of the model being evaluated.
    """
    device = wrapper.device
    val_loss = hyperparameters["val_loss"]

    # TODO -> REVERSE the TRANSFORMATIONS
    # PRINT and STORE the Metrics for both! ?!?!?! so an inital check can be done

    x_test = test_data["x_test"]
    y_test = test_data["y_test"]

    predictions = []
    metrics_results = []

    for i in range(0, x_test.size(0), ts_future):  # Shift by ts_future instead of 1
        x_context = x_test[i, -ts_past:, 0].cpu().numpy()
        y_actual = y_test[i].cpu().numpy()

        x_batch = x_test[i].unsqueeze(0).to(device)
        y_pred = wrapper.predict(x_batch).squeeze(0).cpu().numpy()

        predictions.append((x_context, y_actual, y_pred))

        # Calculate metrics
        metrics = MetricsUtility.calculate_metrics(y_pred, y_actual)
        metrics_results.append(metrics)

        # Generate plot title with model details
        # TODO -> RFACTOR to apply for all MODELS
        plot_title = (
            f"Model: {hyperparameters['config/model_type']}, Val Loss: {val_loss:.4f}\n"
            f"Hidden Size: {hyperparameters.get('config/model_params/hidden_size', 'N/A')}, "
            f"Dropout: {hyperparameters.get('config/model_params/dropout', 'N/A')}, "
            f"Learning Rate: {hyperparameters.get('config/training_params/learning_rate', 'N/A')}"
        )

        # Save plot with model name in filename
        plot_path = output_dir / f"{model_name}_plot_{i + 1}.png"
        plot_predictions(ts_past, ts_future, x_context, y_actual, y_pred, plot_path, plot_title)

    # Save metrics summary
    # TODO -> REFACTOR that metrics_utils already provides correct datatype -> float and not float32
    # TODO -> are the metrics averaged over each snapshot????
    # TODO -> also want maybe from the snapshots the best and worst
    metrics_summary = {
        "average": {metric: float(sum(d[metric] for d in metrics_results) / len(metrics_results)) for metric in metrics_results[0]}
    }
    metrics_path = output_dir / f"{model_name}_metrics_summary.json"
    file_manager.save_json(metrics_summary, metrics_path)
    return metrics_summary


def main():
    # Load configurations and paths
    config_manager = ConfigManager("config.yaml")
    experiment_manager = ExperimentManager(config_manager)

    dataset_config = config_manager.get_dataset_config()
    ts_past = dataset_config["ts_past"]
    ts_future = dataset_config["ts_future"]

    experiment_name, experiment_path, _ = experiment_manager.manage_experiment()

    # Load test data
    data_manager = DataManager(config_manager)
    data_manager.load_raw_data()
    processed_data = data_manager.preprocess_data()
    test_data = {
        "x_test": processed_data["tensor_data"]["x_test"],
        "y_test": processed_data["tensor_data"]["y_test"]
    }

    # TODO -> print test data!!!



    # Load models and evaluate
    trained_models_path = experiment_path / "trained_models"
    output_dir = experiment_path / "evaluation_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_path in trained_models_path.glob("*.pt"):
        model_name = model_path.stem

        # Load hyperparameters from JSON next to the model
        hyperparams_path = model_path.with_name(f"{model_name}_hyperparameters.json")
        if not hyperparams_path.exists():
            raise FileNotFoundError(f"Hyperparameters JSON not found for model: {model_name}")

        hyperparameters = file_manager.load_json(hyperparams_path)

        model_type = hyperparameters["config/model_type"]
        model_params = {
            k.split("/")[-1]: v for k, v in hyperparameters.items() if k.startswith("config/model_params")
        }

        device = get_device()
        model = load_model(model_path, model_type, model_params, device)
        wrapper = WrapperFactory.create_wrapper(model_type, model, None, None, device)

        # Evaluate model
        logger.info(f"Evaluating model: {model_name}")
        evaluate_model(model, wrapper, test_data, ts_past, ts_future, output_dir, hyperparameters, model_name)

if __name__ == "__main__":
    main()
