from typing import *
from transformers import TFAutoModelForSequenceClassification

def from_pretrained(model_name_or_path: Union[str, os.PathLike], *model_args, **kwargs) -> TFAutoModelForSequenceClassification:
    """Loads a pretrained model for sequence classification.

    Args:
        model_name_or_path (Union[str, os.PathLike]): The model name or path to the pretrained model.
        model_args: Additional arguments to pass to the model initialization.
        kwargs: Additional keyword arguments to pass to the model initialization.

    Returns:
        TFAutoModelForSequenceClassification: The loaded pretrained model.
    """
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name_or_path, *model_args, **kwargs)
    return model
