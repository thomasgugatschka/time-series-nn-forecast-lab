import matplotlib.pyplot as plt
import imageio.v2 as imageio

from src.manager.path_manager import path_manager
from src.manager.file_manager import file_manager
from src.manager.env_manager import EnvManager

from src.utilities.logging_utils import get_logger
logger = get_logger(__name__)


class PlotManager:

    def __init__(self, config_manger, data_manager, phase=None):

        self.config_manager = config_manger

        # TODO -> remove the DataManager??? not needed of the prediciton is done outside of the plot_manager
        self.data_manager = data_manager
        self.phase = phase

        self.env_manager = EnvManager()
        self.env_manager.load_env()
        self.plots_show = self.env_manager.get_env_variable('PLOTS_SHOW', default='False', as_bool=True)
        self.plots_save = self.env_manager.get_env_variable('PLOTS_SAVE', default='True', as_bool=True)

        self.main_config = self.config_manager.get_main_config()
        self.dataset_config = self.config_manager.get_dataset_config()

        self.model_type = self.main_config['model_type'].upper()
        self.dataset_name = self.main_config['dataset']
        self.data_params = self.main_config['data_params']

        self.ts_past = self.dataset_config['ts_past']
        self.ts_future = self.dataset_config['ts_future']


        if self.phase is None:
            self.plot_path = path_manager.get_path('experiment')
        else:
            if self.phase == "dataset_visualization":
                self.plot_path = path_manager.get_path_dataset_plots(self.dataset_name)
            else:
                self.plot_path = path_manager.get_path(f"experiment_plots_{self.phase}")

        self.setup_plot_backend()
        self.setup_plot_style()

    def setup_plot_backend(self):
        """
        Dynamically set the Matplotlib backend based on the PLOTS_SHOW environment variable.
        """
        import matplotlib

        if self.plots_show:
            # Use an interactive backend
            backend = self.env_manager.get_env_variable('PLOTS_BACKEND_INTERACTIVE', default='TkAgg')
        else:
            # Use a non-interactive backend
            backend = self.env_manager.get_env_variable('PLOTS_BACKEND_NONINTERACTIVE', default='Agg')

        matplotlib.use(backend)
        logger.info(f"Matplotlib backend set to: {backend}")


    @staticmethod
    def setup_plot_style():
        plt.rcParams['figure.figsize'] = (14, 8)
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['lines.linewidth'] = 2.5

    # TODO -> todo detailed control over legend and stuff -> apply everywhere!!!
    @staticmethod
    def apply_legend(ax, loc='upper right', ncol=1, title=None):
        """
        Apply consistent legend settings.

        Args:
            ax: The Axes object to apply the legend.
            loc: Position of the legend (default is 'upper right').
            ncol: Number of columns in the legend (default is 1).
            title: Title for the legend (optional).
        """
        ax.legend(loc=loc, ncol=ncol, title=title, frameon=True, shadow=False, fancybox=True)

    # TODO -> use this whenever a TEXT is needed in a plot
    @staticmethod
    def apply_text(ax, text, x=0.05, y=0.95, verticalalignment='top', fontsize=10):
        """
        Add consistent text annotations to a plot.

        Args:
            ax: The Axes object to apply the text.
            text: Text string to add.
            x: Horizontal position (default is 0.05).
            y: Vertical position (default is 0.95).
            verticalalignment: Vertical alignment of the text (default is 'top').
            fontsize: Font size of the text (default is 10).
        """
        ax.text(
            x, y, text,
            transform=ax.transAxes,
            verticalalignment=verticalalignment,
            fontsize=fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    def _save_or_show_plot(self, filename):
        """
        Save or show the plot based on environment variables.
        Args:
            filename (str): The name of the file to save the plot.
        """
        try:
            file_path = self.plot_path / filename
            if self.plots_save:
                file_manager.save_plot(plt, file_path)
                logger.info(f"Saved plot to {file_path}")
            if self.plots_show:
                plt.show()
        finally:
            plt.close()  # Ensure resources are released



    # region TRAINING & TUNING PLOTS
    # TODO -> move prediction OUT of plot manager -> should be somewhere else???
    def plot_performance_during_training(self, wrapper, x_val, y_val, epochs, trial_name):
        """
        Plot the performance during training.

        Args:
            wrapper: The model wrapper for making predictions.
            x_val: Validation input data.
            y_val: Validation target data.
            epochs: Current epoch number.
            trial_name: Unique name for the trial.
        """
        # Check if plots are enabled
        if not self.plots_save and not self.plots_show:
            logger.info(f"Skipping plot generation during {self.phase} due to PLOTS_SHOW and PLOTS_SAVE set to False.")
            return

        # Generate predictions and denormalize
        prediction = wrapper.predict(x_val).detach().numpy()
        y_val = y_val.detach().numpy()
        prediction_denorm = self.data_manager.postprocess_data(prediction, selected=True)
        y_val_denorm = self.data_manager.postprocess_data(y_val, selected=True)

        # Plot denormalized values
        plt.plot(y_val_denorm[:, 0], label="Actual")
        plt.plot(prediction_denorm[:, 0], label="Prediction")
        plt.title(f"Denormalized Prediction vs Actual | Epoch: {epochs}")
        plt.legend()

        # Save or show the plot
        filename = f"{trial_name}/epoch_{epochs:04d}.png"
        self._save_or_show_plot(filename)

    def create_gif_from_plots(self, trial_name, gif_name='_training_progress.gif', duration=0.3):
        """
        Creates a GIF from saved plots in the current plot_path.

        Parameters:
        - gif_name: Name of the output GIF file.
        - duration: Duration of each frame in the GIF.
        """
        plots_path = self.plot_path / trial_name

        # Ensure plot_path exists
        if not plots_path.exists():
            logger.warning(f"Plot directory does not exist: {plots_path}")
            return

        # Collect PNG files
        try:
            images = []
            for file_name in sorted(plots_path.iterdir()):
                if file_name.suffix == '.png':
                    images.append(imageio.imread(file_name))

            if not images:
                logger.warning(f"No PNG files found in plot directory: {plots_path}")
                return

            # Output GIF path
            gif_path = plots_path / gif_name

            # Generate GIF
            fps = 1 / duration
            imageio.mimsave(gif_path, images, fps=fps)
        except Exception as e:
            logger.error(f"Failed to create GIF from plots: {e}")
    # endregion


    # region EVALUATION PLOTS

    # TODO -> check if BOTH PLOTs can be merged together!!!! since it is very similar code

    def plot_evaluation(self, ts_past, ts_future, x_context, y_actual, y_predicted, model_name, plot_index, plot_title, denormalized: bool, plot_info=None):
        """
        Plot and save the actual vs. predicted values with context.

        Args:
            ts_past (int): Number of past time steps.
            ts_future (int): Number of future time steps.
            x_context (numpy.ndarray): Context data.
            y_actual (numpy.ndarray): Actual values.
            y_predicted (numpy.ndarray): Predicted values.
            model_name (str): Name of the model being evaluated.
            plot_index (int): Index of the current plot.
            plot_title (str): Title of the plot including model and hyperparameter details.
            plot_info (str): Additional information to display on the plot.
        """
        logger.info(f"Generating evaluation plot for {model_name}, plot index: {plot_index}")

        plt.plot(range(-ts_past, 0), x_context, label="Context", color="blue")
        plt.plot(range(0, ts_future), y_actual, label="Actual", color="green")
        plt.plot(range(0, ts_future), y_predicted, label="Predicted", color="red")
        plt.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
        plt.legend()
        plt.title(plot_title)
        plt.xlabel("Time Steps")
        plt.ylabel("Value")

        if plot_info:
            plt.text(
                0.02, 0.95,
                plot_info,
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment='top'
            )

        if denormalized:
            filename = f"{model_name}_denormalized_plot_{plot_index}.png"
        else:
            filename = f"{model_name}_normalized_plot_{plot_index}.png"

        self._save_or_show_plot(filename)

    def plot_evaluation_full_context(
            self,
            ts_past,
            ts_future,
            x_context,
            y_actual,
            y_predicted,
            model_name,
            plot_index,
            plot_title,
            denormalized: bool,
            plot_info=None,
            all_features_context=None
    ):
        """
        Plot and save the actual vs. predicted values with context.

        Args:
            ts_past (int): Number of past time steps.
            ts_future (int): Number of future time steps.
            x_context (numpy.ndarray): Selected feature context data.
            y_actual (numpy.ndarray): Actual values for the selected feature.
            y_predicted (numpy.ndarray): Predicted values for the selected feature.
            model_name (str): Name of the model being evaluated.
            plot_index (int): Index of the current plot.
            plot_title (str): Title of the plot including model and hyperparameter details.
            denormalized (bool): Flag indicating if the data is denormalized.
            plot_info (str, optional): Additional information to display on the plot.
            all_features_context (numpy.ndarray, optional): Full context data for all features.
        """
        logger.info(f"Generating evaluation plot for {model_name}, plot index: {plot_index}")

        # Plot all input features with lower alpha
        if all_features_context is not None:
            num_features = all_features_context.shape[1]
            for feature_idx in range(num_features):
                plt.plot(
                    range(-ts_past, 0),
                    all_features_context[:, feature_idx],
                    label=f"Feature {feature_idx}" if feature_idx == 0 else None,
                    color="gray",
                    alpha=0.5,
                    linewidth=1,
                )

        # Plot selected feature's context, actual, and predicted values
        plt.plot(range(-ts_past, 0), x_context, label="Selected Feature (Context)", color="blue", linewidth=2)
        plt.plot(range(0, ts_future), y_actual, label="Actual", color="green", linewidth=2)
        plt.plot(range(0, ts_future), y_predicted, label="Predicted", color="red", linewidth=2)
        plt.axvline(x=0, color="black", linestyle="--", linewidth=0.8)

        # Add legend, title, and labels
        plt.legend()
        plt.title(plot_title)
        plt.xlabel("Time Steps")
        plt.ylabel("Value")

        # Add optional info text
        if plot_info:
            plt.text(
                0.02,
                0.95,
                plot_info,
                transform=plt.gca().transAxes,
                fontsize=12,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            )

        # Generate filename
        filename = f"{model_name}_{'denormalized' if denormalized else 'normalized'}_plot_{plot_index}_ALL.png"

        # Save or show the plot
        self._save_or_show_plot(filename)




    def plot_metric_trend(self, metrics, metric_name, title, filename):
        """
        Plot a trend line for a given metric.

        Args:
            metrics (list): List of metric values.
            metric_name (str): Name of the metric.
            title (str): Title for the plot.
            filename (str): File name to save the plot.
        """

        plt.plot(metrics, marker="o", linestyle="-", color="green")
        plt.title(title)
        plt.xlabel("Model Index")
        plt.ylabel(metric_name)
        self._save_or_show_plot(filename)

    def plot_metric_histogram(self, metrics, metric_name, title, filename):
        """
        Plot a histogram for a given metric.

        Args:
            metrics (list): List of metric values.
            metric_name (str): Name of the metric.
            title (str): Title for the plot.
            filename (str): File name to save the plot.
        """
        plt.hist(metrics, bins=10, alpha=0.7, color="blue", edgecolor="black")
        plt.title(title)
        plt.xlabel(metric_name)
        plt.ylabel("Frequency")
        self._save_or_show_plot(filename)

    # endregion


    # region DATASET PLOTS
    def plot_full_dataset(self, df, crop_start, crop_end, metadata):
        """Plot the full dataset with crop markers."""
        dataset_name = self.dataset_name
        selected_feature = metadata["selected_feature"]
        combined_feature_list = metadata["combined_feature_list"]
        sources = self.dataset_config.get("sources", [])

        logger.info("Plotting full dataset.")

        # Plot only the selected feature
        plt.plot(df.iloc[:, selected_feature],
                 label=f"Selected Feature: {combined_feature_list[selected_feature]}", linewidth=2.5)
        plt.axvline(crop_start, color="red", linestyle="--", label="Crop Start")
        plt.axvline(crop_end, color="green", linestyle="--", label="Crop End")
        plt.title(f"Full Dataset ({dataset_name}) - Selected Feature\nCrop Start: {crop_start}, Crop End: {crop_end}")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        plt.annotate(f"Data Source: {sources}\nDatapoints: {len(df)}", xy=(0.02, 0.95), xycoords="axes fraction")
        self._save_or_show_plot(f"{dataset_name}_full_dataset_selected_feature.png")

        # Plot all features
        for i, col in enumerate(df.columns):
            plt.plot(df.iloc[:, i], label=f"{combined_feature_list[i]}")
        plt.axvline(crop_start, color="red", linestyle="--", label="Crop Start")
        plt.axvline(crop_end, color="green", linestyle="--", label="Crop End")
        plt.title(f"Full Dataset ({dataset_name}) - All Features\nCrop Start: {crop_start}, Crop End: {crop_end}")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        self._save_or_show_plot(f"{dataset_name}_full_dataset_all_features.png")

    def plot_cropped_data(self, df, metadata):
        """Plot cropped data with ts_past and ts_future markers."""
        dataset_name = self.dataset_name
        selected_feature = metadata["selected_feature"]
        combined_feature_list = metadata["combined_feature_list"]

        logger.info("Plotting cropped data with ts_past and ts_future markers.")


        plt.plot(df.iloc[:, selected_feature],
                 label=f"Selected Feature: {combined_feature_list[selected_feature]}", linewidth=2.5)
        last_index = df.index[-1]
        cursor_position_ts_past = last_index - self.ts_future - self.ts_past
        cursor_position_ts_future = last_index - self.ts_future

        plt.axvline(cursor_position_ts_past, color="red", linestyle="--", label="ts_past")
        plt.axvline(cursor_position_ts_future, color="green", linestyle="--", label="ts_future")

        plt.title(f"Cropped Dataset ({dataset_name}) - Selected Feature\nData Points: {len(df)}")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        self._save_or_show_plot(f"{dataset_name}_cropped_data.png")

    def plot_data_chunk(self, df, metadata):
        """Plot a chunk of the data with ts_past and ts_future markers."""
        dataset_name = self.dataset_name
        selected_feature = metadata["selected_feature"]
        combined_feature_list = metadata["combined_feature_list"]

        logger.info("Plotting a detailed chunk of data.")
        chunk_size = int((self.ts_past + self.ts_future) * 1.05)  # Include some space around ts_past and ts_future
        start = max(0, len(df) - chunk_size)
        df_chunk = df.iloc[start:]

        plt.figure(figsize=(14, 8))
        plt.plot(df_chunk.iloc[:, selected_feature],
                 label=f"Selected Feature: {combined_feature_list[selected_feature]}", linewidth=2.5)
        last_index = df_chunk.index[-1]
        cursor_position_ts_past = last_index - self.ts_future - self.ts_past
        cursor_position_ts_future = last_index - self.ts_future

        plt.axvline(cursor_position_ts_past, color="red", linestyle="--", label="ts_past")
        plt.axvline(cursor_position_ts_future, color="green", linestyle="--", label="ts_future")

        plt.title(f"Detailed Data Chunk ({dataset_name}) - Selected Feature\nData Points: {len(df_chunk)}")
        plt.xlabel("Time Steps")
        plt.ylabel("Value")
        plt.legend()
        self._save_or_show_plot(f"{dataset_name}_data_chunk.png")

    # endregion