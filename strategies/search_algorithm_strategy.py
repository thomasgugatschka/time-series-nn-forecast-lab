from ray.tune.search.bayesopt import BayesOptSearch
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search.basic_variant import BasicVariantGenerator

class SearchAlgorithmStrategy:
    def __init__(self, algorithm_name="random", metric="val_loss", mode="min"):
        self.algorithm_name = algorithm_name
        self.metric = metric
        self.mode = mode


    # TODO - ADD GRID SEARCH
    def get_search_algorithm(self):
        if self.algorithm_name == "random":
            return BasicVariantGenerator()
        elif self.algorithm_name == "bayesopt":
            return BayesOptSearch(metric=self.metric, mode=self.mode)
        elif self.algorithm_name == "hyperopt":
            return HyperOptSearch(metric=self.metric, mode=self.mode, random_state_seed=42)
        elif self.algorithm_name == "optuna":
            return OptunaSearch(metric=self.metric, mode=self.mode)
        else:
            raise ValueError(f"Unknown search algorithm '{self.algorithm_name}'")

