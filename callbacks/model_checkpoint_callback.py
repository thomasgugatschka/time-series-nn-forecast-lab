import os
import torch
import logging

class ModelCheckpointCallback:
    def __init__(self, checkpoint_dir, monitor='val_loss', mode='min', save_best_only=True):
        self.checkpoint_dir = checkpoint_dir
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        self.counter = 0

        # Ensure checkpoint directory exists
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

    def on_epoch_end(self, epoch, wrapper, val_loss, **kwargs):
        # Determine if the current epoch's validation loss is better than the best score so far
        if self.best_score is None or (
            (self.mode == 'min' and val_loss < self.best_score) or
            (self.mode == 'max' and val_loss > self.best_score)
        ):
            self.best_score = val_loss
            model_save_path = os.path.join(self.checkpoint_dir, f'best_model_epoch_{epoch}.pt')
            torch.save(wrapper.model.state_dict(), model_save_path)
            logging.info(f"Model checkpoint saved at epoch {epoch + 1}: {model_save_path}")

        elif not self.save_best_only:
            model_save_path = os.path.join(self.checkpoint_dir, f'model_epoch_{epoch}.pt')
            torch.save(wrapper.model.state_dict(), model_save_path)
            logging.info(f"Model checkpoint saved at epoch {epoch + 1}: {model_save_path}")
