import torch
import logging
import os
from utilities.data_split_utils import split_and_convert_data
from utilities.plot_utils import plot_prediction, plot_prediction_with_context
from utilities.save_utils import save_parameters_and_results
from models.rnn.rnn_model import RNN
from models.rnn.rnn_wrapper import RNNWrapper
from scripts.preprocessing.data_preprocessing import load_and_preprocess_data
from configs.paths import get_path

# Add RNN class to safe globals
torch.serialization.add_safe_globals([RNN])

def evaluate_model(config_data, config_train, model_name, dataset):
    # Load and preprocess data
    df, x_arr, y_arr, z_mean, z_std, selected_feature, num_features = load_and_preprocess_data(config_data, config_train)

    # Split and convert data
    x_train, x_val, x_test, y_train, y_val, y_test, x_arr_train, x_arr_val, x_arr_test, y_arr_train, y_arr_val, y_arr_test = split_and_convert_data(x_arr, y_arr)

    # Initialize model
    model_params = {
        'num_classes': config_train['ts_future'],
        'input_size': num_features,
        'hidden_size': config_train['hidden_size'],
        'num_layers': config_train['num_layers'],
        'dropout': config_train['dropout']
    }
    best_model = RNN(**model_params)

    # Load trained model state_dict
    trained_model_path = get_path(
        'model_dir', model_name, 'trained_models',
        f'{model_name}_{dataset}_ts{config_train["ts_past"]}-{config_train["ts_future"]}_epochs{config_train["epochs"]}_'
        f'lr{config_train["learning_rate"]}_hs{config_train["hidden_size"]}_nl{config_train["num_layers"]}_dr{config_train["dropout"]}_sf{config_data["station_forecasted"]}_'
        f'feat{selected_feature}.pth'
    )

    logging.info(f"Loading trained model from {trained_model_path}")

    best_model.load_state_dict(torch.load(trained_model_path, map_location=torch.device('cpu'), weights_only=True))
    best_model.eval()

    # Define loss function
    loss_fn = torch.nn.MSELoss()

    # Wrap model for predict
    rnn_wrapper = RNNWrapper(best_model, None, loss_fn)

    # Calculate test loss
    rnn_prediction = rnn_wrapper.predict(x_test)
    rnn_mse_loss = round(loss_fn(rnn_prediction, y_test).item(), 4)
    logging.info(f"{model_name} mse loss: {rnn_mse_loss}")

    # Save training results
    training_results = {
        'STATION_FORECASTED': config_data['station_forecasted'],
        'selected_feature': selected_feature,
        'model_type': model_name,
        'epochs': config_train['epochs'],
        'learning_rate': config_train['learning_rate'],
        'num_classes': config_train['ts_future'],
        'input_size': num_features,
        'hidden_size': config_train['hidden_size'],
        'num_layers': config_train['num_layers'],
        'train_time': None,  # Not available in evaluation
        'rnn_mse_loss': rnn_mse_loss
    }

    # Test model
    rnn_prediction = rnn_wrapper.predict(x_test[-1].unsqueeze(0))
    rnn_prediction = rnn_prediction.detach().numpy()
    rnn_prediction = rnn_prediction * z_std.iloc[selected_feature] + z_mean.iloc[selected_feature]  # Remove z-score normalization using iloc
    rnn_prediction = rnn_prediction[0].tolist()

    # Test target
    test_target = y_arr_test[-1, :]
    test_target = test_target * z_std.iloc[selected_feature] + z_mean.iloc[selected_feature]
    test_target = test_target.tolist()

    # Plot prediction
    plot_prediction(test_target, rnn_prediction, config_data['feature_list'], selected_feature, model_name)

    # Plot prediction and context
    plot_prediction_with_context(df, y_arr_test, rnn_prediction, config_data['feature_list'], selected_feature, model_name, config_train['ts_past'], config_train['ts_future'], z_mean, z_std, training_results)

if __name__ == "__main__":
    from configs.logging_config import setup_logging
    import yaml

    setup_logging()

    dataset = 'weather'
    model_name = 'rnn'

    config_data_file = get_path('config_dir', f'config_data_{dataset}.yaml')
    with open(config_data_file, 'r') as file:
        config_data = yaml.safe_load(file)
    logging.info(f"Loaded configuration from {config_data_file}")
    logging.info(f"Configuration:\n{yaml.dump(config_data, default_flow_style=False)}")

    config_train_file = get_path('config_dir', f'config_train_{model_name}.yaml')
    with open(config_train_file, 'r') as file:
        config_train = yaml.safe_load(file)
    logging.info(f"Loaded configuration from {config_train_file}")
    logging.info(f"Configuration:\n{yaml.dump(config_train, default_flow_style=False)}")

    evaluate_model(config_data, config_train, model_name, dataset)