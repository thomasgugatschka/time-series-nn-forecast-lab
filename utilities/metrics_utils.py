import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MetricsUtility:
    @staticmethod
    def calculate_metrics(predictions, targets):
        """
        Calculate a comprehensive set of error metrics.

        Args:
            predictions (numpy.ndarray or torch.Tensor): Predicted values.
            targets (numpy.ndarray or torch.Tensor): True values.

        Returns:
            dict: A dictionary containing all metrics.
        """
        # Ensure inputs are numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Avoid division by zero for percentage errors
        epsilon = 1e-10
        targets = np.where(targets == 0, epsilon, targets)

        # Core Metrics
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(targets, predictions)
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        smape = np.mean(2 * np.abs(targets - predictions) / (np.abs(targets) + np.abs(predictions))) * 100
        msle = mean_squared_error(np.log1p(targets), np.log1p(predictions))
        rmsle = np.sqrt(msle)

        # Weighted Absolute Percentage Error (WAPE)
        wape = np.sum(np.abs(targets - predictions)) / np.sum(np.abs(targets))

        # Compile results
        return {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "mape": mape,
            "smape": smape,
            "msle": msle,
            "rmsle": rmsle,
            "wape": wape,
        }

    @staticmethod
    def print_metrics(metrics):
        """
        Print metrics in a formatted way.

        Args:
            metrics (dict): A dictionary of metric values.
        """
        print("\nMetrics Summary:")
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")

    @staticmethod
    def get_metric_names():
        """
        Returns a list of available metric names.

        Returns:
            list: Names of all metrics.
        """
        return [
            "mae", "mse", "rmse", "mape", "smape",
            "msle", "rmsle", "wape"
        ]
