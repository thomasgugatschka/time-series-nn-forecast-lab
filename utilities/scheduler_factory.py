from ray.tune.schedulers import ASHAScheduler, FIFOScheduler, HyperBandScheduler


def get_scheduler(scheduler_name, trial_params):
    """
    Factory function to get the scheduler instance.

    Args:
        scheduler_name (str): The name of the scheduler to use.
        trial_params (dict): Dictionary containing parameters like max_epochs, grace_period, etc.

    Returns:
        Scheduler: An instantiated Ray Tune scheduler.
    """
    if scheduler_name == "asha":
        return ASHAScheduler(
            metric="val_loss",
            mode="min",
            max_t=trial_params.get('max_epochs', 50),
            grace_period=trial_params.get('grace_period', 5),
            reduction_factor=trial_params.get('reduction_factor', 2)
        )
    elif scheduler_name == "fifo":
        return FIFOScheduler()
    elif scheduler_name == "hyperband":
        return HyperBandScheduler(
            metric="val_loss",
            mode="min",
            max_t=trial_params.get('max_epochs', 50),
            reduction_factor=trial_params.get('reduction_factor', 3)
        )
    else:
        raise ValueError(f"Unknown scheduler name: {scheduler_name}")
