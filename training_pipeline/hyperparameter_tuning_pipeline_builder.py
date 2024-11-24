from training_pipeline.hyperparameter_tuning_pipeline import HyperparameterTuningPipeline

class HyperparameterTuningPipelineBuilder:
    def __init__(self):
        self.model = None
        self.wrapper = None
        self.data_loaders = None
        self.epochs = None
        self.callback_manager = None

    def set_model(self, model):
        self.model = model
        return self

    def set_wrapper(self, wrapper):
        self.wrapper = wrapper
        return self

    def set_data_loaders(self, train_loader, val_loader, test_loader):
        self.data_loaders = {
            "train": train_loader,
            "val": val_loader,
            "test": test_loader
        }
        return self

    def set_epochs(self, epochs):
        self.epochs = epochs
        return self

    def set_callback_manager(self, callback_manager):
        self.callback_manager = callback_manager
        return self

    def build(self):
        # Check for missing parameters
        missing_params = []
        if not self.model:
            missing_params.append("model")
        if not self.wrapper:
            missing_params.append("wrapper")
        if not self.data_loaders:
            missing_params.append("data_loaders")
        if not self.epochs:
            missing_params.append("epochs")
        if not self.callback_manager:
            missing_params.append("callback_manager")

        if missing_params:
            raise ValueError(f"Missing parameters to build the hyperparameter tuning pipeline: {missing_params}")

        pipeline_config = {
            "model": self.model,
            "data_loaders": self.data_loaders,
            "epochs": self.epochs,
            "callback_manager": self.callback_manager
        }

        # Create and return the HyperparameterTuningPipeline instance
        return HyperparameterTuningPipeline(pipeline_config, self.wrapper)

