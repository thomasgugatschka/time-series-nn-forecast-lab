from callbacks.base_callback import BaseCallback
from utilities.logging_utils import get_logger
logger = get_logger(__name__)


class LoggingCallback(BaseCallback):
    def __init__(self, log_every_n_epochs=1):
        self.log_every_n_epochs = log_every_n_epochs

    def on_epoch_end(self, epoch, wrapper, **kwargs):
        val_loss = kwargs.get('val_loss')
        train_loss = kwargs.get('train_loss')
        if epoch % self.log_every_n_epochs == 0:
            logger.info(f"Epoch {epoch + 1}: Validation Loss = {val_loss:.4f}")
            logger.info(f"Epoch {epoch + 1}: Training Loss = {train_loss:.4f}")
