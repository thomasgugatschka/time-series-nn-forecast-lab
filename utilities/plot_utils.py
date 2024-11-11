import os
import matplotlib.pyplot as plt
import torch

from configs.paths import get_path
from utilities.data_processing_utils import z_score_denormalize

# TODO -> REFACTOR and REMOVE anything MODEL related e.g. rnn_prediction

# TODO -> Refactor the model_name should be in all CAPS

def plot_training_progress(training_loss, validation_loss, model_name):
    """
    Plots the training and validation loss and saves the figure.

    Parameters:
    - training_loss: List of training loss values.
    - validation_loss: List of validation loss values.
    - model_name: Name of the model (used for saving the figure).
    """
    plt.title('Training Progress')
    plt.yscale("log")
    plt.plot(training_loss, label='train')
    plt.plot(validation_loss, label='validation')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    plot_results_directory = get_path('result_dir', model_name)
    figure_path = os.path.join(plot_results_directory, f"Training_progress_{model_name}.png")
    plt.savefig(figure_path)
    plt.show()

def plot_prediction(test_target, rnn_prediction, feature_list, selected_feature, model_name):
    """
    Plots the actual data and rnn predictions and saves the figure.

    Parameters:
    - test_target: Actual data.
    - rnn_prediction: rnn predictions.
    - feature_list: List of features.
    - selected_feature: Index of the selected feature.
    - model_name: Name of the model (used for saving the figure).
    """
    plt.plot(test_target, label="Actual Data")
    plt.plot(rnn_prediction, label="rnn Predictions")
    plt.title('rnn predictions')
    plt.ylabel(f"{feature_list[selected_feature % 5]}")
    plt.xlabel("Time")
    plt.legend()
    plot_results_directory = get_path('result_dir', model_name)
    figure_path = os.path.join(plot_results_directory, f"{model_name}_predictions.png")
    plt.savefig(figure_path)
    plt.show()

def plot_prediction_with_context(df, y_arr_test, rnn_prediction, feature_list, selected_feature, model_name, ts_past, ts_future, z_mean, z_std, new_row):
    """
    Plots the actual data, rnn predictions with context, and saves the figure.

    Parameters:
    - df: DataFrame containing the data.
    - y_arr_test: Array of test target values.
    - rnn_prediction: rnn predictions.
    - feature_list: List of features.
    - selected_feature: Index of the selected feature.
    - model_name: Name of the model (used for saving the figure).
    - TS_PAST: Number of past time steps.
    - TS_FUTURE: Number of future time steps.
    - z_mean: Mean values for normalization.
    - z_std: Standard deviation values for normalization.
    - new_row: Dictionary containing additional information to display on the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.title(f'{model_name} predictions with context')
    start_plot = len(y_arr_test) - 2 * ts_past

    y = df.iloc[:, selected_feature]
    y = y[-len(y_arr_test):]
    y = y * z_std.iloc[selected_feature] + z_mean.iloc[selected_feature]
    a = [x for x in range(start_plot, len(y))]
    plt.plot(a, y[start_plot:], label='Actual data')
    c = [x for x in range(len(y) - ts_future, len(y))]
    plt.plot(c, rnn_prediction, label=f'One-shot multi-step prediction ({ts_future}h)')
    plt.axvline(x=len(y) - ts_future, c='r', linestyle='--')
    plt.ylabel(f"{feature_list[selected_feature % 5]}")
    plt.xlabel("Time")
    plt.legend(loc='upper left')
    plot_results_directory = get_path('result_dir', model_name)
    figure_path = os.path.join(plot_results_directory, f"{model_name}_predictions_context.png")
    text_content = '\n'.join([f'{key}: {value}' for key, value in new_row.items()])
    plt.text(0.05, 0.05, text_content, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8), verticalalignment='bottom', horizontalalignment='left')
    plt.savefig(figure_path)
    plt.show()

def plot_performance_during_training(wrapper, x_val, y_val, z_mean, z_std, selected_feature):
    prediction = wrapper.predict(x_val).detach().numpy()
    prediction_denorm = z_score_denormalize(prediction, z_mean.iloc[selected_feature], z_std.iloc[selected_feature])
    y_val_denorm = z_score_denormalize(y_val.detach().numpy(), z_mean.iloc[selected_feature], z_std.iloc[selected_feature])

    # Plot denormalized values
    plt.figure(figsize=(10, 5))
    plt.plot(y_val_denorm[:, 0], label='Actual')
    plt.plot(prediction_denorm[:, 0], label='Prediction')
    plt.title('Denormalized Prediction vs Actual')
    plt.legend()
    plt.show()