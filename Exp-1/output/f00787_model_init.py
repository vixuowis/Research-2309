from typing import *
from transformers import AutoModelForSequenceClassification

def model_init(trial):
    '''
    Returns an instance of the model for the given trial.

    Args:
        trial: The trial object provided by Optuna.

    Returns:
        An instance of the model for the given trial.
    '''
    return AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
