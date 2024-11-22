import logging

class EarlyStoppingCallback:
    def __init__(self, patience=5, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.best_val_loss = float('inf')
        self.wait = 0
        self.stop_training = False

    def on_epoch_end(self, epoch, wrapper, val_loss, **kwargs):
        # Check if there is an improvement
        if val_loss < self.best_val_loss - self.delta: #TODO - check if this is correct # not sure since it always checks based on best not last...
            self.best_val_loss = val_loss
            self.wait = 0  # Reset the wait counter
        else:
            self.wait += 1
            logging.info(f"No improvement in validation loss for {self.wait} epoch(s). Patience: {self.patience}")

            if self.wait >= self.patience:
                self.stop_training = True
                logging.info(f"Early stopping triggered after {epoch + 1} epochs.")

    def should_stop(self):
        return self.stop_training
