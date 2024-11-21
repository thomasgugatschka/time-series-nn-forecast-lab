import logging

from callbacks.base_callback import BaseCallback
from configs.logging_config import setup_logging

setup_logging()


class LoggingCallback(BaseCallback):
    def __init__(self, log_every_n_epochs=1):
        self.log_every_n_epochs = log_every_n_epochs

    def on_epoch_end(self, epoch, wrapper, val_loss, **kwargs):

        # TODO what about train loss ? -> how to get that?
        if epoch % self.log_every_n_epochs == 0:
            logging.info(f"Epoch {epoch + 1}: Validation Loss = {val_loss:.4f}")
