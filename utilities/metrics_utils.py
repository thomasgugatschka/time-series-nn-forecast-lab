import numpy as np
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(predictions, targets):
    # Ensure the inputs are numpy arrays
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().numpy()



    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)

    mape = np.mean(np.abs((targets - predictions) / targets)) * 100




    return {
        'mae': mae,
        'mse': mse,
        'mape': mape
    }



# TODO -> check references



