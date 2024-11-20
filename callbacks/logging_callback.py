import logging

from callbacks.base_callback import Callback
from configs.logging_config import setup_logging

setup_logging()


class LoggingCallback(Callback):
    def __init__(self, log_every_n_epochs=1):
        self.log_every_n_epochs = log_every_n_epochs

    def on_epoch_end(self, epoch, wrapper, x_val, y_val):
        # Calculate training and validation loss
        train_loss = wrapper.loss_fn(wrapper.model(x_val), y_val).item()  # Training loss for logging
        val_loss = wrapper.loss_fn(wrapper.model(x_val), y_val).item()  # Validation loss for logging

        if epoch % self.log_every_n_epochs == 0:
            logging.info(f'epoch {epoch}. train: {round(train_loss, 4)}, val: {round(val_loss, 4)}')
