import torch
import numpy as np
import random


def get_device():
    """
    Determine and return the appropriate computing device.

    Returns:
        torch.device: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    """
    Set the random seed for reproducibility across various modules.

    Args:
        seed (int): The random seed value to use.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# TODO -> move OPTIMIZER & LOSS function to separeate file in MODELS -> model_utils.py
def get_optimizer(optimizer_name, params, lr):
    """
    Retrieve the specified optimizer.

    Args:
        optimizer_name (str): The name of the optimizer (e.g., 'Adam', 'SGD').
        params (iterable): Parameters of the model to optimize.
        lr (float): Learning rate for the optimizer.

    Returns:
        torch.optim.Optimizer: The selected optimizer instance.

    Raises:
        ValueError: If the optimizer name is invalid.
    """
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
    """
    Retrieve the specified loss function.

    Args:
        loss_function_name (str): Name of the loss function (e.g., 'MSELoss', 'MAELoss').

    Returns:
        torch.nn.Module: The selected loss function instance.

    Raises:
        ValueError: If the loss function name is invalid.
    """

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