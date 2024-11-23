class CallbackManager:
    def __init__(self, callbacks=None):
        self.callbacks = callbacks if callbacks is not None else []

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def on_epoch_end(self, epoch, wrapper, val_loss, **kwargs):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch=epoch, wrapper=wrapper, val_loss=val_loss, **kwargs)
