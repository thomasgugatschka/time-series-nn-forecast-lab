import torch
from ray.air import session

from utilities.logging_utils import get_logger
logger = get_logger(__name__)

class TrainingPipeline:
    """
    Class to manage the training and validation process of a machine learning model.
    """
    def __init__(self, pipeline_config, wrapper):
        """
        Initialize the TrainingPipeline with the given configuration.

        Args:
            pipeline_config (dict): Configuration for the training pipeline.
            wrapper (object): Wrapper for handling model training.
        """
        self.model = pipeline_config["model"]
        self.data_loaders = pipeline_config["data_loaders"]
        self.epochs = pipeline_config["epochs"]
        self.wrapper = wrapper  # Wrapper to handle the training of the model
        self.callback_manager = pipeline_config.get("callback_manager")  # Optional, could be None

    def run(self):
        """
        Execute the full training process, including training and validation.
        """
        try:
            # Trigger training begin callback
            if self.callback_manager:
                self.callback_manager.on_train_begin(wrapper=self.wrapper)

            for epoch in range(self.epochs):
                # Trigger epoch begin callback
                if self.callback_manager:
                    self.callback_manager.on_epoch_begin(epoch=epoch, wrapper=self.wrapper)

                # Train the model for one epoch
                train_loss = self.train_epoch(epoch)

                # Validate the model at the end of the epoch
                val_loss, x_val, y_val = self.validate_epoch(epoch)

                # Trigger epoch end callback
                if self.callback_manager:
                    self.callback_manager.on_epoch_end(
                        epoch=epoch,
                        wrapper=self.wrapper,
                        val_loss=val_loss,
                        train_loss=train_loss,
                        x_val=x_val,
                        y_val=y_val
                    )

                # Check if early stopping or any other stopping condition has been met
                if self.callback_manager and self.callback_manager.should_stop():
                    logger.info(f"Training stopped early at epoch {epoch + 1}")
                    break

                # Report the validation loss to Ray Tune
                session.report({"val_loss": val_loss, "train_loss": train_loss})

        finally:
            if self.callback_manager:
                if 'val_loss' in locals():
                    self.callback_manager.on_train_end(wrapper=self.wrapper, val_loss=val_loss, train_loss=train_loss)
                else:
                    self.callback_manager.on_train_end(wrapper=self.wrapper)


    def train_epoch(self, epoch):
        """
        Train the model for a single epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            float: Average training loss for the epoch.
        """
        train_loss_total = 0
        num_train_batches = 0

        # Train loop over each batch in the training data
        for batch_idx, (x_batch, y_batch) in enumerate(self.data_loaders["train"]):
            # Trigger batch begin callback
            if self.callback_manager:
                self.callback_manager.on_batch_begin(batch=batch_idx, wrapper=self.wrapper)

            x_batch, y_batch = x_batch.to(self.wrapper.device), y_batch.to(self.wrapper.device)
            batch_loss = self.wrapper.train_epoch(x_batch, y_batch)
            train_loss_total += batch_loss
            num_train_batches += 1

            # Trigger batch end callback
            if self.callback_manager:
                self.callback_manager.on_batch_end(batch=batch_idx, wrapper=self.wrapper)

        avg_train_loss = train_loss_total / num_train_batches
        return avg_train_loss

    def validate_epoch(self, epoch):
        """
        Validate the model for a single epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            tuple: Average validation loss and validation data (input and target).
        """
        val_loss_total = 0
        num_val_batches = 0

        x_val_batches = []  # Store x_val batches for callback use
        y_val_batches = []  # Store y_val batches for callback use

        # Trigger validation begin callback
        if self.callback_manager:
            self.callback_manager.on_validation_begin(wrapper=self.wrapper)

        # Validation loop over each batch in the validation data
        for batch_idx, (x_val_batch, y_val_batch) in enumerate(self.data_loaders["val"]):
            x_val_batch, y_val_batch = x_val_batch.to(self.wrapper.device), y_val_batch.to(self.wrapper.device)
            batch_loss = self.wrapper.validate(x_val_batch, y_val_batch)
            val_loss_total += batch_loss
            num_val_batches += 1

            # Append for callback purposes
            x_val_batches.append(x_val_batch.cpu())  # Move to CPU for later usage in callbacks if necessary
            y_val_batches.append(y_val_batch.cpu())  # Same for y_val

        avg_val_loss = val_loss_total / num_val_batches

        # Collect complete validation data for callback usage
        x_val = torch.cat(x_val_batches, dim=0)
        y_val = torch.cat(y_val_batches, dim=0)

        # Trigger validation end callback
        if self.callback_manager:
            self.callback_manager.on_validation_end(val_loss=avg_val_loss, wrapper=self.wrapper)

        return avg_val_loss, x_val, y_val
