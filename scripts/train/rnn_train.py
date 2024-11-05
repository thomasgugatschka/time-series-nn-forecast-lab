import torch
import time
import os
import pandas as pd
import logging
import yaml
from configs.logging_config import setup_logging
from configs.paths import get_path
from utilities.seed_utils import set_seed
from utilities.save_utils import save_parameters_and_results
from utilities.data_split_utils import split_and_convert_data
from utilities.plot_utils import plot_training_progress, plot_prediction, plot_prediction_with_context
from models.rnn.rnn_model import RNN
from models.rnn.rnn_wrapper import RNNWrapper
from scripts.preprocessing.data_preprocessing import load_and_preprocess_data

# region ### INITIAL SETUP ###
set_seed(42)
setup_logging()

dataset = 'weather'
model_name = 'rnn'
# endregion

# region ### LOAD CONFIGURATION ###
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
# endregion

# region ### SET VARIABLES from CONFIGS ###
ts_past = config_train['ts_past']
ts_future = config_train['ts_future']
epochs = config_train['epochs']
dropout = config_train['dropout']
learning_rate = config_train['learning_rate']
hidden_size = config_train['hidden_size']
num_layers = config_train['num_layers']

stations = config_data['stations']  # TODO -> not stations rather "data sources" which depend on each other -> maybe unnecessary complicated...
                                    # TODO -> delete not NEEDED -> remove from all CALCULATIONS!!!!

station_forecasted = config_data['station_forecasted']
feature_list = config_data['feature_list']
feature_forecasted = config_data['feature_forecasted']
# endregion

# region ### LOAD and PREPROCESS DATA ###
df, x_arr, y_arr, z_mean, z_std, selected_feature, num_features \
    = (load_and_preprocess_data(config_data, config_train))
# endregion

# region ### SPLIT and CONVERT DATA ###
x_train, x_val, x_test, y_train, y_val, y_test, x_arr_train, x_arr_val, x_arr_test, y_arr_train, y_arr_val, y_arr_test = split_and_convert_data(x_arr, y_arr)
# endregion

# region ### INITIALIZE MODEL ###
model_params = {
    'num_classes':ts_future,
    'input_size': num_features,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'dropout': dropout
}
model = RNN(**model_params)
logging.info(model)
# endregion

# region ### DEFINE OPTIMIZER and LOSS FUNCTION ###
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss() # TODO - make configurable in config_train
# endregion

# region ### WRAP MODEL for FIT and PREDICT ###
rnn_wrapper = RNNWrapper(model, optimizer, loss_fn)
# endregion

# region ### FIT / TRAIN MODEL ###
model.train() # Set model to training mode
logging.info(f"Start training {model_name.upper()} model")
start_time = time.time()
training_loss, validation_loss = rnn_wrapper.fit(x_train, y_train, x_val, y_val, epochs)
end_time = time.time()
train_time = round(end_time - start_time, 1)
logging.info(f"Execution time: {train_time} seconds")
# endregion

# region ### SAVE TRAINED MODEL ###
best_model = RNN(**model_params)
best_model.load_state_dict(model.state_dict())

trained_model_path = get_path(
    'model_dir', model_name, 'trained_models',
    f'{model_name}_{dataset}_ts{ts_past}-{ts_future}_epochs{epochs}_'
    f'lr{learning_rate}_hs{hidden_size}_nl{num_layers}_dr{dropout}_sf{station_forecasted}_'
    f'feat{selected_feature}.pth'
)
    # _bs{batch_size}_opt{optimizer}_loss{loss_fn} # --> add more parameters to the model name
# Save trained model state_dict
torch.save(best_model.state_dict(), trained_model_path)
logging.info(f"Done training {model_name} model")
# endregion

# region ### PLOT TRAINING PROGRESS ###
plot_training_progress(training_loss, validation_loss, model_name)
# endregion

# region ### CALCUALTE TEST LOSS ###
best_model.eval() # Set model to evaluation mode
rnn_prediction = rnn_wrapper.predict(x_test)
rnn_mse_loss = round(loss_fn(rnn_prediction, y_test).item(), 4)
logging.info(f"{model_name} mse loss: {rnn_mse_loss}")
# endregion

# region ### SAVE TRAINING RESULTS ###
training_results = {
    'STATION_FORECASTED': station_forecasted,
    'selected_feature': selected_feature,
    'model_type': model_name,
    'epochs': epochs,
    'learning_rate': learning_rate,
    'num_classes': ts_future,
    'input_size': num_features,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'train_time': train_time,
    'rnn_mse_loss': rnn_mse_loss
}
save_parameters_and_results(training_results, model_name)
# endregion


# Test model
#rnn_prediction = rnn_wrapper.predict(x_test[-1].unsqueeze(0))
#rnn_prediction = rnn_prediction.detach().numpy()
#rnn_prediction = rnn_prediction * z_std.iloc[selected_feature] + z_mean.iloc[selected_feature]  # Remove z-score normalization using iloc
#rnn_prediction = rnn_prediction[0].tolist()

# Test target
#test_target = y_arr_test[-1, :]
#test_target = test_target * z_std.iloc[selected_feature] + z_mean.iloc[selected_feature]
#test_target = test_target.tolist()


# Plot prediction
#plot_prediction(test_target, rnn_prediction, feature_list, selected_feature, model_name)


# Plot prediction and context
#plot_prediction_with_context(df, y_arr_test, rnn_prediction, feature_list, selected_feature, model_name, ts_past, ts_future, z_mean, z_std, training_results)


