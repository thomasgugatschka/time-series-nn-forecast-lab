import torch
import copy
import logging
from configs.logging_config import setup_logging
from utilities.plot_utils import plot_performance_during_training

setup_logging()

class RNNWrapper:
    def __init__(self, model, optimizer, loss_fn):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn

    def get_model_name(self):
        return self.model.get_model_name()

    # TODO -> remove z_mean, z_std, selected_feature ??? -> MAYBE THERE IS A NICER WAY TO GENERATE THE PLOTS
    #def fit(self, x_train, y_train, x_val, y_val, epochs):
    def fit(self, x_train, y_train, x_val, y_val, epochs, z_mean, z_std, selected_feature):
        training_loss = []
        validation_loss = []
        best_params = None
        min_val_loss = float('inf')

        for epoch in range(epochs):
            # Training step
            self.model.train()
            prediction = self.model(x_train)
            loss = self.loss_fn(prediction, y_train)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Validation step
            self.model.eval()
            with torch.no_grad():
                val_prediction = self.model(x_val)
                val_loss = self.loss_fn(val_prediction, y_val)

            training_loss.append(loss.item())
            validation_loss.append(val_loss.item())

            if val_loss.item() < min_val_loss:
                best_params = copy.deepcopy(self.model.state_dict())
                min_val_loss = val_loss.item()

            if epoch % 1 == 0:
                logging.info(f'epoch {epoch}. train: {round(loss.item(), 4)}, val: {round(val_loss.item(), 4)}')
                #plot_performance_during_training(self, x_val, y_val, z_mean, z_std, selected_feature, epoch)
                plot_performance_during_training(self, x_val, y_val, z_mean, z_std, selected_feature, epoch, self.get_model_name())


        self.model.load_state_dict(best_params)
        return training_loss, validation_loss

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)



