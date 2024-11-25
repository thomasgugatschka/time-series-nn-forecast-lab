import torch
from torch.utils.tensorboard import SummaryWriter
from callbacks.base_callback import BaseCallback
from utilities.logging_utils import get_logger

logger = get_logger(__name__)

class TensorBoardCallback(BaseCallback):
    def __init__(self, log_dir):
        """
        Initialize the TensorBoard callback.

        Args:
            log_dir (str): Directory where TensorBoard logs should be stored.
        """
        super().__init__()
        self.writer = SummaryWriter(log_dir)

    def on_epoch_end(self, epoch, wrapper, **kwargs):
        """
        Called at the end of each epoch to log metrics to TensorBoard.

        Args:
            epoch (int): The current epoch number.
            wrapper (BaseWrapper): The model wrapper.
            **kwargs: Additional arguments.
                - val_loss (float, optional): Validation loss for the current epoch.
                - train_loss (float, optional): Training loss for the current epoch.
        """
        val_loss = kwargs.get('val_loss')
        if val_loss is not None:
            logger.debug(f"Logging validation loss to TensorBoard at epoch {epoch}")
            self.writer.add_scalar('Loss/Validation', val_loss, epoch)
        else:
            logger.warning(f"Validation loss not provided at epoch {epoch}")

        # Log training loss, if available
        train_loss = kwargs.get('train_loss')
        if train_loss is not None:
            logger.debug(f"Logging training loss to TensorBoard at epoch {epoch}")
            self.writer.add_scalar('Loss/Training', train_loss, epoch)

        # Log learning rate, if available
        if wrapper.optimizer is not None:
            for i, param_group in enumerate(wrapper.optimizer.param_groups):
                lr = param_group.get('lr')
                if lr is not None:
                    logger.debug(f"Logging learning rate to TensorBoard for param group {i} at epoch {epoch}")
                    self.writer.add_scalar(f'Learning_Rate/param_group_{i}', lr, epoch)

        # Log model weights and gradients
        for name, param in wrapper.model.named_parameters():
            if param.requires_grad:
                logger.debug(f"Logging weights and gradients for {name} at epoch {epoch}")
                self.writer.add_histogram(f'Weights/{name}', param, epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

    def on_train_end(self, **kwargs):
        """
        Called when the training is complete to close the TensorBoard writer.

        Args:
            **kwargs: Additional arguments.
        """
        logger.info("Training has ended, closing TensorBoard writer.")
        self.writer.close()
