import torch
import logging
import os

from utilities.data_utils import preprocess_data, load_config, split_and_convert_data, z_score_denormalize
from utilities.plot_utils import plot_prediction, plot_prediction_with_context
from utilities.experiment_utils import get_device
from manager.path_manager import path_manager

from models.rnn.rnn_model import RNN
from models.rnn.rnn_wrapper import RNNWrapper



def main(dataset, model_name):

    # region ### INITIAL SETUP ###
    device = get_device()
    logging.info(f"Device: {device}")
    # endregion

    # region ### LOAD CONFIGURATION ###
    config_data = load_config(path_manager.get_path('config_dir', f'config_data_{dataset}.yaml'))
    config_train = load_config(path_manager.get_path('config_dir', f'config_train_{model_name}_{dataset}.yaml'))
    # endregion

    # region ### SET VARIABLES from CONFIGS ###
    ts_past = config_data['ts_past']
    ts_future = config_data['ts_future']
    source_forecasted = config_data['source_forecasted']
    feature_list = config_data['feature_list']

    hidden_size = config_train['hidden_size']
    learning_rate = config_train['learning_rate']
    num_layers = config_train['num_layers']
    dropout = config_train['dropout']
    epochs = config_train['epochs']
    # endregion

    # region ### LOAD and PREPROCESS DATA ###
    df, x_arr, y_arr, z_mean, z_std, selected_feature, num_features = (preprocess_data(config_data, dataset))
    # endregion

    # region ### SPLIT and CONVERT DATA ###
    x_train, x_val, x_test, y_train, y_val, y_test, x_arr_train, x_arr_val, x_arr_test, y_arr_train, y_arr_val, y_arr_test = split_and_convert_data(x_arr, y_arr)
    # endregion

    # region ### INITIALIZE MODEL ###
    model_params = {
        'output_size': ts_future,
        'input_size': num_features,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout
    }
    best_model = RNN(**model_params)
    best_model.to(device)
    logging.info(f"Model initialized with parameters: {model_params}")
    logging.info(best_model)
    # endregion

    # region ### CHECK FOR TRAINED MODEL ###
    trained_model_path = path_manager.get_path(
        'model_dir', model_name, 'trained_models',
        f'{model_name}_{dataset}_ts{ts_past}-{ts_future}_epochs{epochs}_'
        f'lr{learning_rate}_hs{hidden_size}_nl{num_layers}_dr{dropout}_sf{source_forecasted}_'
        f'feat{selected_feature}.pth'
    )

    if not os.path.exists(trained_model_path):
        logging.error(f"No trained model found at {trained_model_path}")
        return
    # endregion

    logging.info(f"Loading trained model from {trained_model_path}")
    best_model.load_state_dict(torch.load(trained_model_path, map_location=device, weights_only=True))
    best_model.eval()

    # Define loss function
    loss_fn = torch.nn.MSELoss()

    # Wrap model for predict
    rnn_wrapper = RNNWrapper(best_model, None, loss_fn)

    # Calculate test loss
    prediction = rnn_wrapper.predict(x_test)
    rnn_mse_loss = round(loss_fn(prediction, y_test).item(), 4)
    logging.info(f"{model_name} mse loss: {rnn_mse_loss}")

    # Test model
    prediction = rnn_wrapper.predict(x_test[-1].unsqueeze(0))
    prediction = prediction.detach().numpy()
    prediction = z_score_denormalize(prediction, z_mean.iloc[selected_feature], z_std.iloc[selected_feature])
    prediction = prediction[0].tolist()

    # Test target
    test_target = y_arr_test[-1, :]
    test_target = z_score_denormalize(test_target, z_mean.iloc[selected_feature], z_std.iloc[selected_feature])
    test_target = test_target.tolist()

    # Plot prediction
    plot_prediction(test_target, prediction, feature_list, selected_feature, model_name)

    # Plot prediction and context
    new_row = {
        'source_forecasted': source_forecasted,
        'selected_feature': selected_feature,
        'model_type': model_name,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'ts_future': ts_future,
        'input_size': num_features,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'rnn_mse_loss': rnn_mse_loss
    }
    plot_prediction_with_context(df, y_arr_test, prediction, feature_list, selected_feature, model_name, ts_past, ts_future, z_mean, z_std, new_row)


if __name__ == "__main__":
    from configs.logging_config import setup_logging

    setup_logging()

    dataset = 'sinusoidal'
    model_name = 'rnn'

    main(dataset, model_name)