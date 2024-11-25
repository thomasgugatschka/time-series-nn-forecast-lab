class BaseCallback:
    def on_train_begin(self, wrapper, **kwargs):
        pass

    def on_train_end(self, wrapper, **kwargs):
        pass

    def on_epoch_begin(self, epoch, wrapper, **kwargs):
        pass

    def on_epoch_end(self, epoch, wrapper, val_loss=None, train_loss=None, **kwargs):
        pass

    def on_batch_begin(self, batch, wrapper, **kwargs):
        pass

    def on_batch_end(self, batch, wrapper, **kwargs):
        pass

    def on_validation_begin(self, wrapper, **kwargs):
        pass

    def on_validation_end(self, wrapper, val_loss=None, **kwargs):
        pass

# TODO add checkpoint saving and loading callbacks

