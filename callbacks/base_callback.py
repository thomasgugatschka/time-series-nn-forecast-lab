class BaseCallback:
    def on_epoch_end(self, epoch, wrapper, val_loss, **kwargs):
        """
        Default implementation of on_epoch_end. This does nothing by default.
        Subclasses can override this method to perform custom actions.
        Args:
            epoch (int): Current epoch number.
            wrapper (BaseWrapper): The model wrapper instance.
            val_loss (float): The validation loss for this epoch.
            **kwargs: Additional arguments that might be needed by the callback.
        """
        pass
