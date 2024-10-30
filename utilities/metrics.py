

# TODO -> check references


# utilities/metrics.py
import numpy as np



from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def calculate_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)