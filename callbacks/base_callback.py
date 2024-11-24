class BaseCallback:
    # TODO -> add more unify the arguements... maybe more **kwargs or add train_loss or move val_loss to kwargs
    def on_epoch_end(self, epoch, wrapper, **kwargs):
        """
        Default implementation of on_epoch_end. This does nothing by default.
        Subclasses can override this method to perform custom actions.
        Args:
            epoch (int): Current epoch number.
            wrapper (BaseWrapper): The model wrapper instance.
            **kwargs: Additional arguments that might be needed by the callback.
        """
        pass
