# File: strategies/strategy_factory.py
from strategies.search_algorithm_strategy import SearchAlgorithmStrategy
from strategies.scheduler_strategy import SchedulerStrategy

class StrategyFactory:
    @staticmethod
    def create_search_algorithm(algorithm_name, metric, mode):
        return SearchAlgorithmStrategy(algorithm_name, metric, mode).get_search_algorithm()

    @staticmethod
    def create_scheduler(scheduler_name, metric, mode, trial_params):
        max_epochs = trial_params.get('max_epochs')
        grace_period = trial_params.get('grace_period')
        reduction_factor = trial_params.get('reduction_factor')

        return SchedulerStrategy(scheduler_name, metric, mode, max_epochs, grace_period,
                                 reduction_factor).get_scheduler()
