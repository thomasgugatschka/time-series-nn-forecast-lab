import torch
from torch.utils.tensorboard import SummaryWriter
from callbacks.base_callback import BaseCallback
from utilities.logging_utils import get_logger

logger = get_logger(__name__)

class TensorBoardCallback(BaseCallback):
    def __init__(self, log_dir, hparams=None, model=None, example_input_tensor=None):
        """
        Initialize the TensorBoard callback.

        Args:
            log_dir (str): Directory where TensorBoard logs should be stored.
            hparams (dict, optional): Hyperparameters to log.
            model (torch.nn.Module, optional): Model to log.
            example_input_tensor (torch.Tensor, optional): Example input for the model.
        """
        super().__init__()
        self.writer = SummaryWriter(log_dir)
        self.hparams = hparams
        self.model = model
        self.example_input = example_input_tensor

    def on_train_begin(self, wrapper, **kwargs):
        """
        Called when the training begins. Logs the model graph and hyperparameters.

        Args:
            wrapper (BaseWrapper): The model wrapper.
            **kwargs: Additional arguments.
        """
        if self.model and self.example_input is not None:
            try:
                # Attempt to add the model graph to TensorBoard
                model_name = wrapper.model.__class__.__name__
                if 'Transformer' not in model_name:
                    logger.debug(f"Logging model graph to TensorBoard for model: {model_name}.")
                    self.writer.add_graph(self.model, self.example_input)
                else:
                    logger.warning(f"Skipping graph logging for {model_name} due to known tracing issues.")
            except (RuntimeError, torch.jit.TracingCheckError, TypeError) as e:
                logger.error(f"Failed to add model graph to TensorBoard for {model_name}. Error: {e}")

        # # Log hyperparameters as well
        # if self.hparams is not None:
        #     metrics = {}  # You can update this with actual metrics if available
        #     try:
        #         logger.debug(f"Logging hyperparameters to TensorBoard.")
        #         self.writer.add_hparams(self.hparams, metrics)
        #     except NotImplementedError as e:
        #         logger.error(f"Failed to add hyperparameters to TensorBoard. Error: {e}")

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

    def on_train_end(self, wrapper, **kwargs):
        """
        Called when the training is complete to log the hyperparameters and close the TensorBoard writer.

        Args:
            wrapper (BaseWrapper): The model wrapper.
            **kwargs: Additional arguments.
                - val_loss (float, optional): Final validation loss.
                - train_loss (float, optional): Final training loss.
        """
        logger.info("Training has ended, logging hyperparameters and closing TensorBoard writer.")

        # Retrieve final validation loss and training loss (or any relevant metric)
        final_val_loss = kwargs.get('val_loss', float('nan'))
        final_train_loss = kwargs.get('train_loss', float('nan'))

        if final_val_loss is float('nan'):
            logger.warning("Final validation loss not provided, defaulting to NaN.")
        if final_train_loss is float('nan'):
            logger.warning("Final training loss not provided, defaulting to NaN.")

        logger.debug(f"Final validation loss: {final_val_loss}")
        logger.debug(f"Final training loss: {final_train_loss}")
        logger.debug(f"Datatype of final validation loss: {type(final_val_loss)}")
        logger.debug(f"Datatype of final training loss: {type(final_train_loss)}")

        # Log hyperparameters and final losses to TensorBoard
        metrics = {
            "final_val_loss": float(final_val_loss),
            "final_train_loss": float(final_train_loss)
        }

        if self.hparams:
            try:
                logger.info("Logging hyperparameters and metrics to TensorBoard.")
                self.writer.add_hparams(self.hparams, metrics)
            except NotImplementedError as e:
                logger.error(f"Failed to add hyperparameters to TensorBoard. Error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error while adding hyperparameters to TensorBoard: {e}")

        # Close TensorBoard writer
        logger.info("Closing TensorBoard writer.")
        self.writer.close()
