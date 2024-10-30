import os
import matplotlib.pyplot as plt
from configs.paths import get_path

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