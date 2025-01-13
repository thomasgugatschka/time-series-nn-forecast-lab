import os
from src.manager.env_manager import EnvManager
from src.utilities.logging_utils import get_logger
logger = get_logger(__name__)

class CallbackManager:
    def __init__(self, callbacks=None):
        """
        Initialize the CallbackManager.

        Args:
            callbacks (list): List of callback instances to manage. Callbacks can be selectively disabled
                             using environment variables with the pattern DISABLE_<CALLBACK_NAME>.
        """
 #       self.callbacks = callbacks if callbacks else []
        self.callbacks = self._filter_callbacks(callbacks if callbacks else [])

    def _filter_callbacks(self, callbacks):
        """
        Filter out disabled callbacks based on environment variables.

        Args:
            callbacks (list): List of callback instances to filter.

        Returns:
            list: Filtered list of callbacks.
        """

        # TODO -> REFACTOR -> Should be integrated solution with the CallbackManager and the BaseCallback
        enabled_callbacks = []
        for callback in callbacks:
            callback_name = type(callback).__name__.upper()
            disable_flag = os.getenv(f"DISABLE_{callback_name}", "False").lower() in ("true", "1", "yes")

            if disable_flag:
                logger.warning(f"Callback {callback_name} is disabled via environment variable.")
            else:
                enabled_callbacks.append(callback)

        return enabled_callbacks

    def on_train_begin(self, wrapper, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_begin'):
                callback.on_train_begin(wrapper=wrapper, **kwargs)

    def on_train_end(self, wrapper, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_end'):
                callback.on_train_end(wrapper=wrapper, **kwargs)

    def on_epoch_begin(self, epoch, wrapper, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_begin'):
                callback.on_epoch_begin(epoch=epoch, wrapper=wrapper, **kwargs)

    def on_epoch_end(self, epoch, wrapper, val_loss=None, train_loss=None, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch=epoch, wrapper=wrapper, val_loss=val_loss, train_loss=train_loss, **kwargs)

    def on_batch_begin(self, batch, wrapper, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_begin'):
                callback.on_batch_begin(batch=batch, wrapper=wrapper, **kwargs)

    def on_batch_end(self, batch, wrapper, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, 'on_batch_end'):
                callback.on_batch_end(batch=batch, wrapper=wrapper, **kwargs)

    def on_validation_begin(self, wrapper, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, 'on_validation_begin'):
                callback.on_validation_begin(wrapper=wrapper, **kwargs)

    def on_validation_end(self, wrapper, val_loss=None, **kwargs):
        for callback in self.callbacks:
            if hasattr(callback, 'on_validation_end'):
                callback.on_validation_end(wrapper=wrapper, val_loss=val_loss, **kwargs)

    def should_stop(self):
        for callback in self.callbacks:
            if hasattr(callback, "should_stop") and callback.should_stop():
                return True
        return False
