from typing import *
from ray import tune

def ray_hp_space(trial):
    """Returns a dictionary representing the hyperparameter space for Ray Tune.

    Args:
        trial (Trial): The trial object that represents the current trial.

    Returns:
        dict: A dictionary representing the hyperparameter space."""
    return {
        "learning_rate": tune.loguniform(1e-6, 1e-4),
        "per_device_train_batch_size": tune.choice([16, 32, 64, 128]),
    }
