from typing import *
def sigopt_hp_space(trial):
    """Define the hyperparameter search space, different backends need different format.

    For sigopt, see sigopt [object_parameter](https://docs.sigopt.com/ai-module-api-references/api_reference/objects/object_parameter), it's like following:"""
    return [
        {"bounds": {"min": 1e-6, "max": 1e-4}, "name": "learning_rate", "type": "double"},
        {
            "categorical_values": ["16", "32", "64", "128"],
            "name": "per_device_train_batch_size",
            "type": "categorical",
        },
    ]
