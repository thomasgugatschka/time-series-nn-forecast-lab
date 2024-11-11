import os
import pandas as pd
from configs.paths import get_path

def save_parameters_and_results(new_row, model_name):
    plot_results_directory = get_path('result_dir', model_name)
    csv_file_path = os.path.join(plot_results_directory, "parameter_iterations.csv")

    # TODO -> change / refactor --> not STATION instead SOURCE
    if not os.path.isfile(csv_file_path):
        df_params_iter = pd.DataFrame(columns=['STATION_FORECASTED', 'selected_feature', 'model_type', 'epochs', 'learning_rate', 'num_classes', 'input_size', 'hidden_size', 'num_layers', 'train_time', 'rnn_mse_loss'])
    else:
        df_params_iter = pd.read_csv(csv_file_path)

    if df_params_iter.empty or df_params_iter.isna().all().all():
        df_params_iter = pd.DataFrame([new_row])
    else:
        df_params_iter = pd.concat([df_params_iter, pd.DataFrame([new_row])], ignore_index=True)

    df_params_iter.to_csv(csv_file_path, index=False)