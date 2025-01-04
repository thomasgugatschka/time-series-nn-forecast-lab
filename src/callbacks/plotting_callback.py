from src.callbacks.base_callback import BaseCallback
from src.utilities.logging_utils import get_logger

logger = get_logger(__name__)

class PlottingCallback(BaseCallback):
    """
    A callback for generating plots during training and creating a summary GIF of model performance.

    Args:
        plot_manager (PlotManager): Instance of PlotManager for handling plot creation.
        trial_name (str): Unique name for the trial, used to organize plots and GIFs.
    """
    def __init__(self, plot_manager, trial_name):
        super().__init__()
        self.plot_manager = plot_manager
        self.trial_name = trial_name


    def on_epoch_end(self, epoch, wrapper, **kwargs):
        """
        Called at the end of each epoch to visualize the model's performance on validation data.

        Args:
            epoch (int): The current epoch number.
            wrapper (BaseWrapper): The model wrapper handling training and predictions.
            **kwargs: Additional arguments that may include:
                - x_val (torch.Tensor, optional): Validation input features.
                - y_val (torch.Tensor, optional): Validation targets.

        Notes:
            If validation data is not provided, plotting will be skipped, and a warning is logged.
        """
        x_val = kwargs.get('x_val')
        y_val = kwargs.get('y_val')

        if x_val is not None and y_val is not None:
            logger.info(f"Plotting model performance at epoch {epoch}")
            self.plot_manager.plot_performance_during_training(wrapper, x_val, y_val, epoch, self.trial_name)
        else:
            logger.warning(f"Validation data is missing for epoch {epoch}, skipping plotting.")

    def on_train_end(self, wrapper, **kwargs):
        """
        Called at the end of training to generate a summary GIF from the saved plots.

        Args:
            wrapper (BaseWrapper): The model wrapper.
            **kwargs: Additional arguments (unused in this method).
        """
        self.plot_manager.create_gif_from_plots(self.trial_name)