import argparse
import numpy as np
import torch

from pathlib import Path

from src.manager.experiment_manager import ExperimentManager
from src.manager.plot_manager import PlotManager
from src.manager.config_manager import ConfigManager
from src.manager.data_manager import DataManager
from src.models.model_factory import ModelFactory
from src.models.wrapper_factory import WrapperFactory
from src.manager.file_manager import file_manager
from src.manager.path_manager import path_manager
from src.utilities.experiment_utils import get_device
from src.utilities.metrics_utils import MetricsUtility

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


def evaluate_model(wrapper, hyperparameters, test_data, ts_past, ts_future, data_manager, model_name, plot_manager):
    """
    Evaluate a single model on the test dataset.

    Args:
        wrapper: Wrapper for handling model operations.
        hyperparameters (dict): Hyperparameters used to train the model.
        test_data (dict): Test data containing input and target tensors.
        ts_past (int): Number of past time steps.
        ts_future (int): Number of future time steps.
        data_manager (DataManager): Instance for handling data.
        model_name (str): Name of the model.
        plot_manager (PlotManager): Instance for handling plots.
    """
    device = wrapper.device
    val_loss = hyperparameters["val_loss"]

    x_test = test_data["x_test"]
    y_test = test_data["y_test"]

    raw_metrics_norm = []
    raw_metrics_denorm = []

                            # TODO is the , 0] the index of the feature ??? -> need to use the selected_feature
    selected_feature_idx = 0  # TODO - Replace with the actual index of the selected feature

    for i in range(0, x_test.size(0), ts_future):  # Shift by ts_future instead of 1


        # Extract the context for all features
        x_context_all_features = x_test[i, -ts_past:, :].cpu().numpy()  # Shape: [ts_past, num_features]

        # Extract the selected feature from the context # TODO - not sure if needed, since original calcualtion works
        x_context = x_context_all_features[:, selected_feature_idx]  # Shape: [ts_past]



        x_context = x_test[i, -ts_past:, selected_feature_idx].cpu().numpy()
        y_actual = y_test[i].cpu().numpy()

        x_batch = x_test[i].unsqueeze(0).to(device)
        y_pred = wrapper.predict(x_batch).squeeze(0).cpu().numpy()

        # Reverse transformations
        y_actual_denorm = data_manager.postprocess_data(y_actual, selected=True)
        y_pred_denorm = data_manager.postprocess_data(y_pred, selected=True)
        x_context_denorm = data_manager.postprocess_data(x_context, selected=True)

        # Reverse transformations for all features
        x_context_denorm_all_features = data_manager.postprocess_data(x_context_all_features, selected=False)

        # Calculate metrics
        metrics_norm = MetricsUtility.calculate_metrics(y_pred, y_actual)
        MetricsUtility.print_metrics(metrics_norm)
        metrics_denorm = MetricsUtility.calculate_metrics(y_pred_denorm, y_actual_denorm)
        MetricsUtility.print_metrics(metrics_denorm)

        raw_metrics_norm.append(metrics_norm)
        raw_metrics_denorm.append(metrics_denorm)



        # Generate a generic info text for plot
        model_params = {
            k.split("/")[-1]: v for k, v in hyperparameters.items() if k.startswith("config/model_params")
        }
        training_params = {
            k.split("/")[-1]: v for k, v in hyperparameters.items() if k.startswith("config/training_params")
        }
        model_param_str = ", ".join([f"{k}: {v}" for k, v in model_params.items()])
        training_param_str = ", ".join([f"{k}: {v}" for k, v in training_params.items()])
        info_text = (
            f"Model: {hyperparameters['config/model_type']}, Val Loss: {val_loss:.4f}\n"
            f"{model_param_str}\n{training_param_str}"
        )

        # TODO - make all calls the same like -> plot_evaluation_full_context

        # Generate plot(s)
        plot_title = f"Model: {hyperparameters['config/model_type']}, Val Loss: {val_loss:.4f} (Normalized)"
        plot_manager.plot_evaluation(
            ts_past, ts_future, x_context, y_actual, y_pred, model_name, i + 1, plot_title, denormalized=False, plot_info=info_text)

        plot_title = f"Model: {hyperparameters['config/model_type']}, Val Loss: {val_loss:.4f} (Denormalized)"
        plot_manager.plot_evaluation(
            ts_past, ts_future, x_context_denorm, y_actual_denorm, y_pred_denorm, model_name, i + 1, plot_title, denormalized=True, plot_info=info_text)

        plot_title = f"Model Evaluation (Denormalized) | Sample {i + 1}"
        plot_manager.plot_evaluation_full_context(
            ts_past=ts_past,
            ts_future=ts_future,
            x_context=x_context_denorm,
            y_actual=y_actual_denorm,
            y_predicted=y_pred_denorm,
            model_name=model_name,
            plot_index=i + 1,
            plot_title=plot_title,
            denormalized=True,
            plot_info=info_text,
            all_features_context=x_context_denorm_all_features,
        )


    # Aggregate metrics
    def aggregate_metrics(metrics, comparison_metric="mse"):
        """
        Aggregate metrics across evaluation snapshots.
        Args:
            metrics (list): List of dictionaries containing metric values for each snapshot.
            comparison_metric (str): Metric to use for determining the "best" and "worst" snapshots.
        Returns:
            dict: Aggregated metrics including average, stddev, best, and worst.
        """
        if not metrics:
            raise ValueError("Metrics list is empty.")

        if comparison_metric not in metrics[0]:
            raise KeyError(f"Comparison metric '{comparison_metric}' is not in the metrics dictionary.")

        # Calculate averages
        averages = {metric: float(sum(d[metric] for d in metrics) / len(metrics)) for metric in metrics[0]}

        # Calculate standard deviations
        stddevs = {
            metric: float((sum((d[metric] - averages[metric]) ** 2 for d in metrics) / len(metrics)) ** 0.5)
            for metric in metrics[0]
        }

        # Determine best and worst snapshots based on the comparison metric
        best = min(metrics, key=lambda x: x[comparison_metric])
        worst = max(metrics, key=lambda x: x[comparison_metric])

        return {"average": averages, "stddev": stddevs, "best": best, "worst": worst}

    metrics_summary = {
        "normalized": aggregate_metrics(raw_metrics_norm),
        "denormalized": aggregate_metrics(raw_metrics_denorm),
        "raw_metrics_norm": raw_metrics_norm,  # Store raw metrics
        "raw_metrics_denorm": raw_metrics_denorm,  # Store raw metrics
    }

    # Convert metrics to JSON-serializable format
    metrics_summary = convert_to_python_types(metrics_summary)

    # Save metrics summary
    metrics_path = path_manager.get_path("experiment_evaluation_results") / f"{model_name}_metrics_summary.json"
    file_manager.save_json(metrics_summary, metrics_path)
    logger.info(f"Metrics summary saved to {metrics_path}")

    return metrics_summary


def aggregate_overall_metrics(all_metrics, model_metadata):
    """
    Aggregate metrics across all models and include metadata.

    Args:
        all_metrics (list): List of metrics summaries for all models.
        model_metadata (list): List of metadata for each model.

    Returns:
        dict: Overall summary with aggregated metrics and metadata.
    """
    def aggregate(metrics, key, subkey="average"):
        all_values = [model_metrics[key][subkey] for model_metrics in metrics if key in model_metrics]
        averages = {metric: float(sum(d[metric] for d in all_values) / len(all_values)) for metric in all_values[0]}
        stddevs = {
            metric: float((sum((d[metric] - averages[metric]) ** 2 for d in all_values) / len(all_values)) ** 0.5)
            for metric in all_values[0]
        }
        best = min(all_values, key=lambda x: x.get("mae", float("inf")))
        worst = max(all_values, key=lambda x: x.get("mae", float("-inf")))

        return {"average": averages, "stddev": stddevs, "best": best, "worst": worst}

    return {
        "metadata": model_metadata,
        "normalized": aggregate(all_metrics, key="normalized"),
        "denormalized": aggregate(all_metrics, key="denormalized"),
        "raw_metrics_norm": [model_metrics.get("raw_metrics_norm", []) for model_metrics in all_metrics],
        "raw_metrics_denorm": [model_metrics.get("raw_metrics_denorm", []) for model_metrics in all_metrics],
    }





# TODO - CHECK if possible to make metric configurable... currently mix between mse and mae....
# TODO -> both or all are needed ?


def plot_overall_metrics(plot_manager, overall_metrics):
    """
    Visualize the overall metrics using histograms and trend lines.

    Args:
        plot_manager (PlotManager): Instance for handling plots.
        overall_metrics (dict): Overall metrics summary.
    """
    # Validate structure
    if "denormalized" not in overall_metrics or "normalized" not in overall_metrics:
        logger.error("Overall metrics summary does not contain 'denormalized' or 'normalized' keys.")
        return

    # 1. Plot histogram for MAE (denormalized raw metrics)
    mae_values = [
        snapshot["mae"]
        for model_raw_metrics in overall_metrics["raw_metrics_denorm"]
        for snapshot in model_raw_metrics
        if "mae" in snapshot
    ]
    plot_manager.plot_metric_histogram(
        mae_values,
        metric_name="MAE",
        title="MAE Distribution (Denormalized)",
        filename="mae_histogram.png",
    )

    # 2. Plot histogram for MSE (denormalized raw metrics)
    mse_values = [
        snapshot["mse"]
        for model_raw_metrics in overall_metrics["raw_metrics_denorm"]
        for snapshot in model_raw_metrics
        if "mse" in snapshot
    ]
    plot_manager.plot_metric_histogram(
        mse_values,
        metric_name="MSE",
        title="MSE Distribution (Denormalized)",
        filename="mse_histogram.png",
    )

    # 3. Plot trend for MAE (denormalized average metrics)
    if "average" in overall_metrics["denormalized"]:
        average_mae = overall_metrics["denormalized"]["average"]["mae"]
        stddev_mae = overall_metrics["denormalized"]["stddev"]["mae"]
        plot_manager.plot_metric_trend(
            [average_mae],  # Pass as a list to maintain compatibility
            metric_name="MAE",
            title=f"Mean Absolute Error Trend (Denormalized)\nStddev: {stddev_mae:.4f}",
            filename="mae_trend.png",
        )

    # 4. Plot trend for MSE (denormalized average metrics)
    if "average" in overall_metrics["denormalized"]:
        average_mse = overall_metrics["denormalized"]["average"]["mse"]
        stddev_mse = overall_metrics["denormalized"]["stddev"]["mse"]
        plot_manager.plot_metric_trend(
            [average_mse],  # Pass as a list to maintain compatibility
            metric_name="MSE",
            title=f"Mean Squared Error Trend (Denormalized)\nStddev: {stddev_mse:.4f}",
            filename="mse_trend.png",
        )


def convert_to_python_types(data):
    """
    Recursively convert numpy data types to Python-native types.

    Args:
        data: The input data (dict, list, or scalar).

    Returns:
        The converted data with Python-native types.
    """
    if isinstance(data, dict):
        return {k: convert_to_python_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_to_python_types(item) for item in data]
    elif isinstance(data, (np.integer, int)):
        return int(data)
    elif isinstance(data, (np.floating, float)):
        return float(data)
    else:
        return data


def main():
    # Load configurations and paths
    args = parse_args()
    config_manager = ConfigManager(args.config_path)
    data_manager = DataManager(config_manager)
    experiment_manager = ExperimentManager(config_manager)
    experiment_name, experiment_path, _ = experiment_manager.manage_experiment()
    plot_manager = PlotManager(config_manager, data_manager, phase="evaluation")

    dataset_config = config_manager.get_dataset_config()
    ts_past = dataset_config["ts_past"]
    ts_future = dataset_config["ts_future"]

    # Load test data
    data_manager.load_raw_data()
    processed_data = data_manager.preprocess_data()
    test_data = {
        "x_test": processed_data["tensor_data"]["x_test"],
        "y_test": processed_data["tensor_data"]["y_test"],
    }

    logger.info(f"Test Data Shape: x_test: {test_data['x_test'].shape}, y_test: {test_data['y_test'].shape}")

    # TODO -> Print test_data to see what it looks like

    # TODO -> add val_loss and stuff to the final metrics ?!?!?!?

    # Load models and evaluate
    trained_models_path = path_manager.get_path("experiment_trained_models")
    all_metrics = []
    model_metadata = []

    for model_path in trained_models_path.glob("*.pt"):
        model_name = model_path.stem

        # Load hyperparameters from JSON next to the model
        hyperparams_path = model_path.with_name(f"{model_name}_hyperparameters.json")
        if not hyperparams_path.exists():
            raise FileNotFoundError(f"Hyperparameters JSON not found for model: {model_name}")

        hyperparameters = file_manager.load_json(hyperparams_path)

        model_metadata.append({
            "model_name": model_name,
            "hyperparameters": hyperparameters,
            "metrics_summary_path": str(
                path_manager.get_path("experiment_evaluation_results") / f"{model_name}_metrics_summary.json")
        })

        model_type = hyperparameters["config/model_type"]
        model_params = {
            k.split("/")[-1]: v for k, v in hyperparameters.items() if k.startswith("config/model_params")
        }

        device = get_device()
        model = load_model(model_path, model_type, model_params, device)
        wrapper = WrapperFactory.create_wrapper(model_type, model, None, None, device)

        # Evaluate model
        logger.info(f"Evaluating model: {model_name}")
        metrics_summary = (
            evaluate_model(wrapper, hyperparameters, test_data, ts_past, ts_future, data_manager, model_name, plot_manager))

        all_metrics.append(metrics_summary)

        # Aggregate overall metrics
        overall_metrics = aggregate_overall_metrics(all_metrics, model_metadata)

        # Save overall metrics summary
        overall_metrics_path = path_manager.get_path("experiment_evaluation_results") / "overall_metrics_summary.json"
        file_manager.save_json(overall_metrics, overall_metrics_path)
        logger.info(f"Overall metrics summary saved to {overall_metrics_path}")

        # Optional: Visualize overall metrics
        plot_overall_metrics(plot_manager, overall_metrics)

if __name__ == "__main__":
    main()
