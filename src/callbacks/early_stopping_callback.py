from src.callbacks.base_callback import BaseCallback
from src.utilities.logging_utils import get_logger
logger = get_logger(__name__)

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, patience=5, delta=0.01):
        super().__init__()
        self.patience = patience
        self.delta = delta
        self.best_val_loss = float('inf')
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, wrapper, val_loss=None, **kwargs):
        if val_loss is None:
            return # No validation loss to compare

        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
            self.wait = 0  # Reset the wait counter
        else:
            self.wait += 1
            logger.info(f"No improvement in validation loss for {self.wait} epoch(s). Patience: {self.patience}")
            logger.info(f"Current best validation loss: {self.best_val_loss:.4f}")

            if self.wait >= self.patience:
                self.stop_training = True
                logger.info(f"Early stopping triggered after {epoch} epochs.")

    def should_stop(self):
        return self.stop_training
