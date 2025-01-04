from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.basic_variant import BasicVariantGenerator

class SearchAlgorithmStrategy:
    def __init__(self, algorithm_name="random", metric="val_loss", mode="min"):
        self.algorithm_name = algorithm_name
        self.metric = metric
        self.mode = mode

    def get_search_algorithm(self):
        algorithms = {
            "random": BasicVariantGenerator(),
            "bayesopt": BayesOptSearch(metric=self.metric, mode=self.mode),
            "hyperopt": HyperOptSearch(metric=self.metric, mode=self.mode, random_state_seed=42),
            "optuna": OptunaSearch(metric=self.metric, mode=self.mode)
        }
        return algorithms.get(self.algorithm_name) or self._raise_unknown_algorithm()

    def _raise_unknown_algorithm(self):
        raise ValueError(f"Unknown search algorithm '{self.algorithm_name}'")
