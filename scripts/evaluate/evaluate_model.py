import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from manager.config_manager import ConfigManager
from manager.path_manager import path_manager
from models.model_factory import ModelFactory
from utilities.logging_utils import setup_logging, get_logger
from metrics_utility import MetricsUtility
from manager.data_manager import DataManager

# Initialize logging
setup_logging()
logger = get_logger(__name__)

def load_model(model_path, model_type, model_params, device):
    """
    Load the trained model from a checkpoint.

    Args:
        model_path (str): Path to the model checkpoint.
        model_type (str): Type of the model (e.g., RNN, LSTM).
        model_params (dict): Parameters for the model architecture.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded model.
    """
    model = ModelFactory.create_model(model_type, **model_params).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Model {model_type} loaded from {model_path}.")
    return model

def evaluate_model(model, data_loader, device):
    """
    Evaluate the model on a test dataset.

    Args:
        model (torch.nn.Module): Trained model.
        data_loader (DataLoader): DataLoader for the test dataset.
        device (torch.device): Device for inference.

    Returns:
        tuple: Predictions and targets as numpy arrays.
    """
    predictions, targets = [], []

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            predictions.append(y_pred.cpu().numpy())
            targets.append(y_batch.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    logger.info("Evaluation completed.")
    return predictions, targets

def main(config_path, model_path, batch_size, num_batches):
    """
    Main function to load the model, evaluate it, and compute metrics.

    Args:
        config_path (str): Path to the configuration file.
        model_path (str): Path to the trained model checkpoint.
        batch_size (int): Batch size for the DataLoader.
        num_batches (int): Number of batches to use from the test set.
    """
    # Load configuration
    config_manager = ConfigManager(config_path)
    dataset_config = config_manager.get_dataset_config()
    model_type = config_manager.get("model_type")
    model_params = config_manager.get("model_params")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load test data
    data_manager = DataManager(config_manager)
    data_manager.load_raw_data()
    test_data = data_manager.get_split("test")
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Load the trained model
    model = load_model(model_path, model_type, model_params, device)

    # Evaluate the model
    predictions, targets = evaluate_model(model, test_loader, device)

    # Calculate metrics
    metrics = MetricsUtility.calculate_metrics(predictions, targets)
    MetricsUtility.print_metrics(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a test dataset.")
    parser.add_argument("--config_path", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model checkpoint.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for the DataLoader.")
    parser.add_argument("--num_batches", type=int, default=None, help="Number of batches to use from the test set.")
    args = parser.parse_args()

    main(args.config_path, args.model_path, args.batch_size, args.num_batches)
