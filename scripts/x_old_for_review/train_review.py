import torch
import time
import logging

from configs.logging_config import setup_logging
from manager.path_manager import get_path

from utilities.experiment_utils import get_device
from utilities.experiment_utils import set_seed
from utilities.save_utils import save_parameters_and_results
from utilities.data_utils import preprocess_data, split_and_convert_data, z_score_denormalize, load_config
from utilities.plot_utils import plot_training_progress, plot_prediction, plot_prediction_with_context, create_gif_from_plots

from models.model_factory import model_factory
from models.wrapper_factory import wrapper_factory



def main(dataset, model_name):

    # region ### INITIAL SETUP ###
    set_seed(42)
    setup_logging()
    device = get_device()
    logging.info(f"Device: {device}")
    # endregion

    # region ### LOAD CONFIGURATION ###
    config_data = load_config(get_path('config_dir', f'config_data_{dataset}.yaml'))
    config_train = load_config(get_path('config_dir', f'config_train_{model_name}_{dataset}.yaml'))
    # endregion

    # region ### SET VARIABLES from CONFIGS ###
    epochs = config_train['epochs']
    dropout = config_train['dropout']
    learning_rate = config_train['learning_rate']
    hidden_size = config_train['hidden_size']
    num_layers = config_train['num_layers']

    sources = config_data['sources']
    source_forecasted = config_data['source_forecasted']
    feature_list = config_data['feature_list']
    feature_forecasted = config_data['feature_forecasted']
    ts_past = config_data['ts_past']
    ts_future = config_data['ts_future']
    # endregion

    # region ### LOAD and PREPROCESS DATA ###
    df, x_arr, y_arr, z_mean, z_std, selected_feature, num_features = (preprocess_data(config_data, dataset))
    # endregion

    # region ### SPLIT and CONVERT DATA ###
    x_train, x_val, x_test, y_train, y_val, y_test, x_arr_train, x_arr_val, x_arr_test, y_arr_train, y_arr_val, y_arr_test = split_and_convert_data(x_arr, y_arr)
    # endregion

    # region ### INITIALIZE MODEL ###
    model_params = {
        'output_size':ts_future,
        'input_size': num_features,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'dropout': dropout
    }

    model = model_factory(model_name, **model_params)
    model.to(device) # GPU or CPU
    logging.info(model)
    # endregion

    # region ### DEFINE OPTIMIZER and LOSS FUNCTION ###
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss() # TODO - make configurable in config_train
    # endregion

    # region ### WRAP MODEL for FIT and PREDICT ###
    #rnn_wrapper = RNNWrapper(model, optimizer, loss_fn)
    wrapper = wrapper_factory(model_name, model, optimizer, loss_fn)
    # endregion

    # region ### FIT / TRAIN MODEL ###
    model.train() # Set model to training mode
    logging.info(f"Start training {model_name.upper()} model")
    start_time = time.time()
    training_loss, validation_loss = wrapper.fit(x_train, y_train, x_val, y_val, epochs, z_mean, z_std, selected_feature)
    end_time = time.time()
    train_time = round(end_time - start_time, 1)
    logging.info(f"Execution time: {train_time} seconds")
    # endregion

    # region ### SAVE TRAINED MODEL ###
    trained_model_path = get_path(
        'model_dir', model_name, 'trained_models',
        f'{model_name}_{dataset}_ts{ts_past}-{ts_future}_epochs{epochs}_'
        f'lr{learning_rate}_hs{hidden_size}_nl{num_layers}_dr{dropout}_sf{source_forecasted}_'
        f'feat{selected_feature}.pth'
    )
        # _bs{batch_size}_opt{optimizer}_loss{loss_fn} # --> add more parameters to the model name
    # Save trained model state_dict
    torch.save(model.state_dict(), trained_model_path)
    logging.info(f"Done training {model_name} model")
    # endregion

    # region ### PLOT TRAINING PROGRESS ###
    plot_training_progress(training_loss, validation_loss, model_name)
    # endregion

    # region ### CALCUALTE TEST LOSS ###
    model.eval() # Set model to evaluation mode
    prediction = wrapper.predict(x_test)
    rnn_mse_loss = round(loss_fn(prediction, y_test).item(), 4)
    logging.info(f"{model_name} mse loss: {rnn_mse_loss}")
    # endregion

    # region ### SAVE TRAINING RESULTS ###
    training_results = {
        'source_forecasted': source_forecasted,
        'selected_feature': selected_feature,
        'model_type': model_name,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'ts_future': ts_future,
        'input_size': num_features,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'train_time': train_time,
        'rnn_mse_loss': rnn_mse_loss
    }
    save_parameters_and_results(training_results, model_name)

    create_gif_from_plots(model_name)
    # endregion


    # Test model
    prediction = wrapper.predict(x_test[-1].unsqueeze(0))
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
    plot_prediction_with_context(df, y_arr_test, prediction, feature_list, selected_feature, model_name, ts_past, ts_future, z_mean, z_std, training_results)


if __name__ == "__main__":

    #dataset = 'weather_station'
    dataset = 'sinusoidal'
    model_name = 'rnn'

    main(dataset, model_name)