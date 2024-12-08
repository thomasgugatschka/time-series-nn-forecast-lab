class CallbackManager:
    def __init__(self, callbacks=None):
        self.callbacks = callbacks if callbacks else []

# TODO -> move CallbackManager to callback folder

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
