import psutil

from src.callbacks.base_callback import BaseCallback
from src.utilities.logging_utils import get_logger

logger = get_logger(__name__)

class LoggingCallback(BaseCallback):
    def __init__(self, log_every_n_epochs=1):
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_train_begin(self, wrapper, **kwargs):
        logger.info("Training has started.")

    def on_train_end(self, wrapper, **kwargs):
        #logger.info(f"Final Memory usage: {psutil.virtual_memory()}")
        #logger.info(f"Final CPU usage: {psutil.cpu_percent(interval=1)}")
        logger.info("Training has ended.")

    def on_epoch_begin(self, epoch, wrapper, **kwargs):
        logger.info(f"Starting epoch {epoch}.")
        #logger.info(f"Memory usage before epoch {epoch + 1}: {psutil.virtual_memory()}")
        #logger.info(f"CPU usage before epoch {epoch + 1}: {psutil.cpu_percent(interval=1)}")

    def on_epoch_end(self, epoch, wrapper, val_loss=None, train_loss=None, **kwargs):
        if epoch % self.log_every_n_epochs == 0:
            if val_loss is not None and train_loss is not None:
                logger.info(f"Epoch {epoch} || Validation Loss = {val_loss:.4f} || Training Loss = {train_loss:.4f}")
                #logger.info(f"Memory usage after epoch {epoch + 1}: {psutil.virtual_memory()}")
                #logger.info(f"CPU usage after epoch {epoch + 1}: {psutil.cpu_percent(interval=1)}")
            else:
                logger.info(f"Epoch {epoch}: No validation or training loss recorded.")


    def on_batch_begin(self, batch, wrapper, **kwargs):
        logger.debug(f"Starting batch {batch}.")

    def on_batch_end(self, batch, wrapper, **kwargs):
        logger.debug(f"Finished batch {batch}.")

    def on_validation_begin(self, wrapper, **kwargs):
        logger.info("Validation phase has started.")

    def on_validation_end(self, wrapper, val_loss=None, **kwargs):
        logger.info("Validation phase has ended.")
