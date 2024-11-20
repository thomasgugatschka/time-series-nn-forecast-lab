import torch
import copy

class RNNWrapper:
    def __init__(self, model, optimizer, loss_fn, callbacks=None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.callbacks = callbacks if callbacks is not None else []

    def get_model_name(self):
        return self.model.get_model_name()

    def fit(self, x_train, y_train, x_val, y_val, epochs):
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

            # Trigger callbacks at the end of the epoch
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, self, x_val, y_val)

        self.model.load_state_dict(best_params)
        return training_loss, validation_loss


    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            return self.model(x)



