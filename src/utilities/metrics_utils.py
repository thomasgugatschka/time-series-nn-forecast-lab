import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.utilities.logging_utils import get_logger
logger = get_logger(__name__)

class MetricsUtility:
    """
    Utility class for calculating and managing error metrics.
    """

    @staticmethod
    def calculate_metrics(predictions, targets):
        """
        Calculate a comprehensive set of error metrics between predictions and targets.
        Handles edge cases such as division by zero and negative values for logarithmic metrics.

        Args:
            predictions (numpy.ndarray or torch.Tensor): Predicted values from the model.
            targets (numpy.ndarray or torch.Tensor): Actual target values.

        Returns:
            dict: A dictionary containing the following metrics:
                - mae: Mean Absolute Error.
                - mse: Mean Squared Error.
                - rmse: Root Mean Squared Error.
                - mape: Mean Absolute Percentage Error.
                - smape: Symmetric Mean Absolute Percentage Error.
                - msle: Mean Squared Logarithmic Error (if no negative values are detected).
                - rmsle: Root Mean Squared Logarithmic Error (if no negative values are detected).
                - wape: Weighted Absolute Percentage Error.
                - corr: Correlation coefficient.
                - nrmse: Normalized RMSE (normalized by the range of target values).
                - rrse: Root Relative Squared Error.
        """

        # Ensure inputs are numpy arrays
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.detach().cpu().numpy()

        # Avoid division by zero for percentage errors
        epsilon = 1e-10
        targets = np.where(targets == 0, epsilon, targets)

        # Check for negative values
        has_negatives = np.any(predictions < 0) or np.any(targets < 0)

        # Core Metrics
        # MSE - Mean Squared Error: Range [0, inf), lower is better
        mse = mean_squared_error(targets, predictions)
        # RMSE - Root Mean Squared Error: Same range as MSE but in original scale, lower is better
        rmse = np.sqrt(mse)
        # MAE - Mean Absolute Error: Range [0, inf), lower is better
        mae = mean_absolute_error(targets, predictions)
        # MAPE - Mean Absolute Percentage Error: [0, inf), lower is better
        mape = np.mean(np.abs((targets - predictions) / targets)) * 100
        # sMAPE - Symmetric MAPE: [0, 200], lower is better
        smape = np.mean(2 * np.abs(targets - predictions) / (np.abs(targets) + np.abs(predictions))) * 100
        # WAPE - Weighted Absolute Percentage Error: [0, 1], lower is better
        wape = np.sum(np.abs(targets - predictions)) / np.sum(np.abs(targets))

        # Log-based metrics (only calculated if there are no negative values)
        if not has_negatives:
            # MSLE - Mean Squared Log Error: [0, inf), lower is better
            msle = mean_squared_error(np.log1p(targets), np.log1p(predictions))
            # RMSLE - Root Mean Squared Log Error: Same range as MSLE, lower is better
            rmsle = np.sqrt(msle)
        else:
            logger.warning("Negative values detected. Skipping MSLE and RMSLE computation.")
            msle = float('nan')
            rmsle = float('nan')


        # CORR - Correlation Coefficient: Range [-1, 1], closer to 1 indicates stronger positive correlation
        corr = np.corrcoef(targets.flatten(), predictions.flatten())[0, 1] if len(targets) > 1 else float('nan')

        # NRMSE - Normalized RMSE: Range [0, inf), lower is better. Normalized by the target range.
        nrmse = rmse / (np.max(targets) - np.min(targets)) if np.max(targets) != np.min(targets) else float('nan')

        # RRSE - Root Relative Squared Error: Range [0, inf), lower is better. Compares to a mean prediction baseline.
        mean_target = np.mean(targets)
        rrse = np.sqrt(np.sum((targets - predictions) ** 2) / np.sum((targets - mean_target) ** 2)) if len(
            targets) > 1 else float('nan')

        # Compile results
        return {
            "mae": mae,  # Lower is better
            "mse": mse,  # Lower is better
            "rmse": rmse,  # Lower is better
            "mape": mape,  # Lower is better
            "smape": smape,  # Lower is better
            "msle": msle,  # Lower is better
            "rmsle": rmsle,  # Lower is better
            "wape": wape,  # Lower is better
            "corr": corr,  # Closer to 1 is better
            "nrmse": nrmse,  # Lower is better
            "rrse": rrse,  # Lower is better
        }

    @staticmethod
    def print_metrics(metrics):
        """
        Print calculated metrics in a formatted manner.

        Args:
            metrics (dict): Dictionary of calculated metrics.
        """

        logger.info("Metrics Summary:")
        for metric, value in metrics.items():
            logger.info(f"\t{metric.upper()}: {value:.4f}")

