import os
import torch
from src.callbacks.base_callback import BaseCallback
from src.utilities.logging_utils import get_logger

logger = get_logger(__name__)

class ModelCheckpointCallback(BaseCallback):
    def __init__(self, checkpoint_dir, monitor='val_loss', mode='min', save_best_only=True):
        """
        Initialize the Model Checkpoint Callback.

        Args:
            checkpoint_dir (str): Directory to save model checkpoints.
            monitor (str): Metric to monitor for saving the model.
            mode (str): 'min' or 'max'. Whether to save the model when the monitored metric decreases or increases.
            save_best_only (bool): If True, only saves the model when the monitored metric improves.
        """
        super().__init__()
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def on_epoch_end(self, epoch, wrapper, val_loss=None, **kwargs):
        """
        Called at the end of each epoch to save model checkpoints based on validation loss.

        Args:
            epoch (int): The current epoch number.
            wrapper (BaseWrapper): The model wrapper.
            val_loss (float): The validation loss at the end of the epoch.
            **kwargs: Additional arguments.
        """
        # Only proceed if validation loss is provided
        if val_loss is None:
            logger.warning(f"Validation loss not provided at epoch {epoch}. Skipping model checkpoint.")
            return

        # Determine if the current epoch's validation loss is better than the best score so far
        if self.best_score is None or (
            (self.mode == 'min' and val_loss < self.best_score) or
            (self.mode == 'max' and val_loss > self.best_score)
        ):
            self.best_score = val_loss
            model_save_path = os.path.join(self.checkpoint_dir, f'best_model_epoch_{epoch + 1}.pt')
            torch.save(wrapper.model.state_dict(), model_save_path)
            logger.info(f"Best model checkpoint saved at epoch {epoch + 1}: {model_save_path}")

        # Save every epoch's model if `save_best_only` is False
        elif not self.save_best_only:
            model_save_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch + 1}.pt')
            torch.save(wrapper.model.state_dict(), model_save_path)
            logger.info(f"Model checkpoint saved at epoch {epoch + 1}: {model_save_path}")
