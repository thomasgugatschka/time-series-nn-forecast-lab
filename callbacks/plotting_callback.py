from callbacks.base_callback import BaseCallback
from utilities.plot_utils import plot_performance_during_training
from utilities.logging_utils import get_logger

logger = get_logger(__name__)

class PlottingCallback(BaseCallback):
    def __init__(self, data_manager):
        """
        Initialize the PlottingCallback.

        Args:
            z_mean: Mean values for denormalizing the data.
            z_std: Standard deviation values for denormalizing the data.
            selected_feature: The index of the selected feature for plotting.
        """
        super().__init__()
        self.data_manager = data_manager

    def on_epoch_end(self, epoch, wrapper, **kwargs):
        """
        Called at the end of each epoch to plot the model's performance.

        Args:
            epoch (int): The current epoch number.
            wrapper (BaseWrapper): The model wrapper.
            **kwargs: Additional arguments that may include:
                - x_val (torch.Tensor, optional): Validation input features.
                - y_val (torch.Tensor, optional): Validation targets.
        """
        x_val = kwargs.get('x_val')
        y_val = kwargs.get('y_val')

        if x_val is not None and y_val is not None:
            logger.info(f"Plotting model performance at epoch {epoch}")
            # Import the plot utility here to avoid circular imports if needed
            plot_performance_during_training(wrapper, x_val, y_val, epoch, wrapper.get_model_name(), self.data_manager)
        else:
            logger.warning(f"Validation data is missing for epoch {epoch}, skipping plotting.")
