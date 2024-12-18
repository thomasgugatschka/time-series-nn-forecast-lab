import torch
import numpy as np
import random

# TODO write comment for everything in here

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(optimizer_name, params, lr):
    """Retrieve the appropriate optimizer."""
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
    """Retrieve the appropriate loss function."""
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