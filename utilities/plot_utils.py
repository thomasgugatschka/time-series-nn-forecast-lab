import os
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from utilities.data_utils import z_score_denormalize

from manager.path_manager import path_manager
from manager.file_manager import file_manager

from utilities.logging_utils import get_logger
logger = get_logger(__name__)


# TODO -> REFACTOR and REMOVE anything MODEL related e.g. rnn_prediction

# TODO -> Refactor the model_name should be in all CAPS

# Function to set up the plot styles globally
def setup_plot_style():
    """
    Set global settings for plots to maintain consistency across all plots.
    """
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['lines.linewidth'] = 2.5

# Call the setup_plot_style function when this module is imported
setup_plot_style()


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

    results_directory = path_manager.get_path_result(model_name)
    figure_path = results_directory / f"Training_progress_{model_name}.png"
    file_manager.save_plot(plt, figure_path)


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

    results_directory = path_manager.get_path_result(model_name)
    figure_path = results_directory / f"{model_name}_predictions.png"
    file_manager.save_plot(plt, figure_path)



def plot_prediction_with_context(df, y_arr_test, rnn_prediction, feature_list, selected_feature, model_name, ts_past,
                                 ts_future, z_mean, z_std, new_row):
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
    plt.figure(figsize=(14, 8))
    plt.title(f'{model_name} predictions with context')
    start_plot = len(y_arr_test) - int((ts_past + ts_future)*1.05)

    y = df.iloc[:, selected_feature]
    y = y[-len(y_arr_test):]
    y = y * z_std.iloc[selected_feature] + z_mean.iloc[selected_feature]
    a = [x for x in range(start_plot, len(y))]
    plt.plot(a, y[start_plot:], label='Actual data')
    c = [x for x in range(len(y) - ts_future, len(y))]
    plt.plot(c, rnn_prediction, label=f'One-shot multi-step prediction ({ts_future}h)')

    plt.axvline(x=len(y) - ts_future, c='green', linestyle='--')
    plt.axvline(x=len(y) - ts_future - ts_past, c='red', linestyle='--')

    #plt.ylim(-12.5 , 12.5)

    plt.ylabel(f"{feature_list[selected_feature % 5]}")
    plt.xlabel("Time")
    plt.legend(loc='upper left')
    plot_results_directory = path_manager.get_path('result_dir', model_name)

    text_content = '\n'.join([f'{key}: {value}' for key, value in new_row.items()])
    plt.text(0.05, 0.05, text_content, transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8),
             verticalalignment='bottom', horizontalalignment='left')

    results_directory = path_manager.get_path_result(model_name)
    figure_path = results_directory / f"{model_name}_predictions_context.png"
    file_manager.save_plot(plt, figure_path)


def plot_performance_during_training(wrapper, x_val, y_val, z_mean, z_std, selected_feature, epochs, model_name):
    # TODO - modify... and store with some sort of logic th pngs.... because now everything is in the same folder so no idendification at all
    prediction = wrapper.predict(x_val).detach().numpy()
    prediction_denorm = z_score_denormalize(prediction, z_mean.iloc[selected_feature], z_std.iloc[selected_feature])
    y_val_denorm = z_score_denormalize(y_val.detach().numpy(), z_mean.iloc[selected_feature],
                                       z_std.iloc[selected_feature])

    # Plot denormalized values
    plt.figure(figsize=(14, 8))
    plt.plot(y_val_denorm[:, 0], label='Actual')
    plt.plot(prediction_denorm[:, 0], label='Prediction')
    plt.title('Denormalized Prediction vs Actual | Epoch: ' + str(epochs))
    plt.legend()
    plot_results_directory = path_manager.get_path_training_plot(model_name=model_name)
    os.makedirs(plot_results_directory, exist_ok=True)
    figure_path = os.path.join(plot_results_directory, f"epoch_{epochs:04d}.png")
    plt.savefig(figure_path)

    # Show the plot (comment out if you don't want to see the plots during training)
    plt.show()

    plt.close()  # Close the plot to ensure it is rendered correctly


def create_gif_from_plots(model_name, gif_name='training_progress.gif', duration=0.3):
    """
    Creates a GIF from saved plots.

    Parameters:
    - model_name: Name of the model (used for locating the plots).
    - gif_name: Name of the output GIF file.
    - duration: Duration of each frame in the GIF.
    """
    plot_results_directory = path_manager.get_path_training_plot(model_name=model_name)
    images = []
    # TODO - refactor GIF is NOT handled at ALL
    gif_path = path_manager.get_path('result_dir', model_name, gif_name)

    for file_name in sorted(os.listdir(plot_results_directory)):
        if file_name.endswith('.png'):
            file_path = os.path.join(plot_results_directory, file_name)
            images.append(imageio.imread(file_path))

    fps = 1 / duration

    imageio.mimsave(gif_path, images, fps=fps)

    logger.info(f"GIF saved at {gif_path}")


if __name__ == "__main__":
    create_gif_from_plots('rnn')

