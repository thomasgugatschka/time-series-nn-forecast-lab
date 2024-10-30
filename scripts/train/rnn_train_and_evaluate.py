import torch
import sys
import copy
import time
import matplotlib.pyplot as plt
import os
import pandas as pd
import logging
import numpy as np
import random
from configs.logging_config import setup_logging
from utilities.utilities import split_sequences
from configs.paths import get_path
from utilities.plotting.plot_training_progress import plot_training_progress
from models.RNN.rnn_model import RNN
from models.RNN.rnn_wrapper import RNNWrapper

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# <editor-fold desc="Setup">
set_seed(42)
setup_logging()
logging.info("Start training RNN model")
# </editor-fold>




# Station and feature that we want to forecast
stations = [30165,48374,49608]
STATION_FORECASTED = 0
feature_list = ['Temp (degC)', 'Rel Hum (%)', 'Precip. Amount (mm)', 'Stn Press (kPa)', 'Wind Spd (km/h)']
FEATURE_FORECASTED = 1
selected_feature = len(feature_list) * STATION_FORECASTED + FEATURE_FORECASTED
num_features = len(stations) * len(feature_list)

# Context and forecast length#
TS_PAST = 120   # Time steps to look into the past (context) [h]
TS_FUTURE = 24  # Time steps to look into the future (forecast) [h]

# Global nn parameters
chosen_model = "RNN"              # Choose which model structure to use
epochs = 10                      # Training epochs
dropout = .0                      # Dropout
learning_rate = 0.005             # Learning rate
num_classes = TS_FUTURE
input_size = num_features
hidden_size = 32
num_layers = 2
output_dim = TS_FUTURE

# Import time-series from stored pickles
df_list = []
files = os.listdir(get_path('data_dir', 'pickles'))
for file in files:
    target_file = get_path('data_dir', 'pickles', file)
    df = pd.read_pickle(target_file)
    df = df.drop(columns='Date/Time (LST)')
    df_list.append(df)

nb_stations = len(files)
df = pd.concat(df_list, axis=1)
df = df.iloc[40000:60000, :]  # crop df to lower calc time

# Preprocessing data with z-score normalization
z_mean = df.mean()
z_std = df.std()
df = (df - z_mean) / z_std

# Feed context and prediction results to the model (adding another dimension)
x_arr, y_arr = split_sequences(df, df.iloc[:, selected_feature], TS_PAST, TS_FUTURE)

# Datasplit properties
frac_train = 0.7
frac_val = 0.15
num_samples = x_arr.shape[0]
num_train = int(frac_train * num_samples)
num_val = int(frac_val * num_samples)
num_test = num_samples - num_train - num_val

# Split the data into train/validate/test
x_arr_train = x_arr[:num_train, :, :]
x_arr_val = x_arr[num_train:num_train + num_val, :, :]
x_arr_test = x_arr[num_train + num_val:, :, :]
print(f"\nx_arr shapes:\n{x_arr_train.shape}\n{x_arr_val.shape}\n{x_arr_test.shape}")
y_arr_train = y_arr[:num_train, :]
y_arr_val = y_arr[num_train:num_train + num_val, :]
y_arr_test = y_arr[num_train + num_val:, :]
print(f"\ny_arr shapes:\n{y_arr_train.shape}\n{y_arr_val.shape}\n{y_arr_test.shape}")

# Tensor shape should be x_train(12264,120,15) and y_train(12264,24)
x_train = torch.tensor(x_arr_train, dtype=torch.float32)
x_val = torch.tensor(x_arr_val, dtype=torch.float32)
x_test = torch.tensor(x_arr_test, dtype=torch.float32)
y_train = torch.tensor(y_arr_train, dtype=torch.float32)
y_val = torch.tensor(y_arr_val, dtype=torch.float32)
y_test = torch.tensor(y_arr_test, dtype=torch.float32)
train_len = x_train.size()[0]
print(f"\nTensor shapes (x):\n{x_train.shape}\n{x_val.shape}\n{x_test.shape}")
print(f"\nTensor shapes (y):\n{y_train.shape}\n{y_val.shape}\n{y_test.shape}\n")






# Initialize model
model_params = {
    'num_classes': num_classes,
    'input_size': input_size,
    'hidden_size': hidden_size,
    'num_layers': num_layers,
    'dropout': dropout,
}
model = RNN(**model_params)

logging.info(model)

# Define optimizer and loss function
optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

# Wrap the model
rnn_wrapper = RNNWrapper(model, optimizer, loss_fn)

# Fit the model
start_time = time.time()
training_loss, validation_loss = rnn_wrapper.fit(x_train, y_train, x_val, y_val, epochs)
end_time = time.time()

# Calculate the elapsed time
train_time = round(end_time - start_time, 1)
logging.info(f"Execution time: {train_time} seconds")

# Plot training progress
plot_training_progress(training_loss, validation_loss, 'RNN')

# Load best trained model
best_model = RNN(**model_params)
best_model.load_state_dict(model.state_dict())
trained_model_path = get_path('model_dir', 'RNN', 'trained_models',f'rnn_trained_model_{epochs}_{FEATURE_FORECASTED}.pt')
torch.save(best_model, trained_model_path)
best_model.eval()

# Calculate test loss
rnn_prediction = rnn_wrapper.predict(x_test)
rnn_mse_loss = round(loss_fn(rnn_prediction, y_test).item(), 4)
logging.info(f"RNN mse loss: {rnn_mse_loss}") #TODO - change to to variable instead of RNN

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
plt.plot(test_target, label="Actual Data")
plt.plot(rnn_prediction, label="RNN Predictions")
plt.title('RNN predictions')
plt.ylabel(f"{feature_list[selected_feature % 5]}")
plt.xlabel("Time")
plt.legend()

plot_results_directory = get_path('result_dir', 'RNN')

figure_path = os.path.join(plot_results_directory, "RNN_predictions.png")
plt.savefig(figure_path)
plt.show()

# Save parameters and results to csv
csv_file_path = os.path.join(plot_results_directory, "parameter_iterations.csv")
if not os.path.isfile(csv_file_path):
    df_params_iter = pd.DataFrame(columns=['STATION_FORECASTED', 'selected_feature', 'model_type', 'epochs', 'learning_rate', 'num_classes', 'input_size', 'hidden_size', 'num_layers', 'train_time', 'rnn_mse_loss'])
else:
    df_params_iter = pd.read_csv(csv_file_path)
new_row = {'STATION_FORECASTED': STATION_FORECASTED,
           'selected_feature': selected_feature,
           'model_type': chosen_model,
           'epochs': epochs,
           'learning_rate': learning_rate,
           'num_classes': num_classes,
           'input_size': input_size,
           'hidden_size': hidden_size,
           'num_layers': num_layers,
           'train_time': train_time,
           'rnn_mse_loss': rnn_mse_loss}
df_params_iter = pd.concat([df_params_iter, pd.DataFrame([new_row])], ignore_index=True)
df_params_iter.to_csv(csv_file_path, index=False)

# Plot prediction and context
plt.figure(figsize=(10, 6))
plt.title('RNN predictions with context')
start_plot = len(y_arr_test) - 2 * TS_PAST
y = df.iloc[:, selected_feature]
y = y[-len(y_arr_test):]
y = y * z_std.iloc[selected_feature] + z_mean.iloc[selected_feature]
a = [x for x in range(start_plot, len(y))]
plt.plot(a, y[start_plot:], label='Actual data')
c = [x for x in range(len(y) - TS_FUTURE, len(y))]
plt.plot(c, rnn_prediction, label=f'One-shot multi-step prediction ({TS_FUTURE}h)')
plt.axvline(x=len(y) - TS_FUTURE, c='r', linestyle='--')
plt.ylabel(f"{feature_list[selected_feature % 5]}")
plt.xlabel("Time")
plt.legend(loc='upper left')
figure_path = os.path.join(plot_results_directory, "RNN_predictions_context.png")
text_content = '\n'.join([f'{key}: {value}' for key, value in new_row.items()])
plt.text(0.05, 0.05, text_content, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom', horizontalalignment='left')
plt.savefig(figure_path)
plt.show()

print("Done")