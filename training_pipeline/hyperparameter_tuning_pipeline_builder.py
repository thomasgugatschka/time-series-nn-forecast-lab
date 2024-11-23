from strategies.search_algorithm_strategy import SearchAlgorithmStrategy
from ray.tune.schedulers import ASHAScheduler

class TrainingPipelineBuilder:
    def __init__(self):
        self.model = None
        self.search_algorithm = None
        self.scheduler = None
        self.data_loaders = None
        self.epochs = None

    def set_model(self, model):
        self.model = model
        return self

    def set_search_algorithm(self, algorithm_name, metric="val_loss", mode="min"):
        strategy = SearchAlgorithmStrategy(algorithm_name, metric, mode)
        self.search_algorithm = strategy.get_search_algorithm()
        return self

    # TODO - check is the scheduler even possible to set from a config file?
    def set_scheduler(self, max_epochs, grace_period, reduction_factor):
        self.scheduler = ASHAScheduler(
            metric="val_loss",
            mode="min",
            max_t=max_epochs,
            grace_period=grace_period,
            reduction_factor=reduction_factor
        )
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

    def build(self):
        if not (self.model and self.search_algorithm and self.scheduler and self.data_loaders and self.epochs):
            raise ValueError("Missing parameters to build training pipeline")
        return {
            "model": self.model,
            "search_algorithm": self.search_algorithm,
            "scheduler": self.scheduler,
            "data_loaders": self.data_loaders,
            "epochs": self.epochs
        }
