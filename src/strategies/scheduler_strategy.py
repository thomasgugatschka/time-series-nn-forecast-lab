from ray.tune.schedulers import ASHAScheduler
from ray.tune.schedulers import FIFOScheduler

class SchedulerStrategy:
    def __init__(self, scheduler_name="asha", metric="val_loss", mode="min", max_epochs=100, grace_period=5, reduction_factor=2):
        self.scheduler_name = scheduler_name
        self.metric = metric
        self.mode = mode
        self.max_epochs = max_epochs
        self.grace_period = grace_period
        self.reduction_factor = reduction_factor

    def get_scheduler(self):
        schedulers = {
            "asha": ASHAScheduler(
                metric=self.metric,
                mode=self.mode,
                max_t=self.max_epochs,
                grace_period=self.grace_period,
                reduction_factor=self.reduction_factor
            ),
            "fifo": FIFOScheduler()
        }
        return schedulers.get(self.scheduler_name) or self._raise_unknown_scheduler()

    def _raise_unknown_scheduler(self):
        raise ValueError(f"Unknown scheduler '{self.scheduler_name}'")
