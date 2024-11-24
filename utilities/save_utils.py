import os
import pandas as pd
from manager.path_manager import path_manager

from utilities.logging_utils import get_logger
logger = get_logger(__name__)

def save_parameters_and_results(new_row, model_name):
    csv_file_path = os.path.join(path_manager.get_path('result_dir', model_name), "parameter_iterations.csv")

    # Check if the file exists and is not empty
    if os.path.exists(csv_file_path) and os.path.getsize(csv_file_path) > 0:
        df_params_iter = pd.read_csv(csv_file_path)
    else:
        df_params_iter = pd.DataFrame()

    # Append new results
    df_params_iter = pd.concat([df_params_iter, pd.DataFrame([new_row])], ignore_index=True)

    # Save the updated DataFrame back to the CSV file
    df_params_iter.to_csv(csv_file_path, index=False)



def update_results_in_csv(csv_file_path, chosen_model, feature_list, FEATURE_FORECASTED, model_mse_loss, model_mae_loss, model_r2_loss):
    if os.path.isfile(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        columns = ['Feature'] + [f"{chosen_model.replace(' ', '_')}_{metric}" for metric in ['MSE', 'MAE', 'R2']]
        df = pd.DataFrame(columns=columns)
        df['Feature'] = feature_list

    chosen_model_no_space = chosen_model.replace(' ', '_')
    chosen_feature_index = FEATURE_FORECASTED

    df.at[chosen_feature_index, f"{chosen_model_no_space}_MSE"] = round(model_mse_loss, 4)
    df.at[chosen_feature_index, f"{chosen_model_no_space}_MAE"] = round(model_mae_loss, 4)
    df.at[chosen_feature_index, f"{chosen_model_no_space}_R2"] = round(model_r2_loss, 4)

    df.to_csv(csv_file_path, index=False)
    logger.info(f"Results for {chosen_model} and {feature_list[chosen_feature_index]} updated in {csv_file_path}")