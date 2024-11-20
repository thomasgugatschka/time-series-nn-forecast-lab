from callbacks.base_callback import Callback
from utilities.plot_utils import plot_performance_during_training


class PlottingCallback(Callback):
    def __init__(self, z_mean, z_std, selected_feature):
        self.z_mean = z_mean
        self.z_std = z_std
        self.selected_feature = selected_feature

    def on_epoch_end(self, epoch, wrapper, x_val, y_val):
        # Import the plot utility here to avoid circular imports if needed
        plot_performance_during_training(wrapper, x_val, y_val, self.z_mean, self.z_std, self.selected_feature, epoch, wrapper.get_model_name())
