from typing import *
from transformers import DistilBertModel


def from_pretrained(model_name_or_path: Optional[str] = None, config: Optional[Union[str, PretrainedConfig]] = None, cache_dir: Optional[str] = None, from_tf: bool = False, *model_args, **kwargs) -> PreTrainedModel:
    '''
    Load a pretrained model from a given model identifier or path.

    Args:
        model_name_or_path (:obj:`str`, `optional`, defaults to `None`): The model identifier or the path to a pretrained model archive containing:

            - `pytorch_model.bin` file (containing the weights)
            - `config.json` file (containing the model configuration)

        config (:obj:`PretrainedConfig`, `optional`, defaults to `None`): An optional configuration for the model to use instead of an automatically loaded configuation. Configuration can be automatically loaded when:

            - The model is a model provided by the library (loaded with the `shortcut-name` string of a pretrained model).
            - The model was saved using `save_pretrained('./test/saved_model/')` and is reloaded by suppling the save directory.

        cache_dir (:obj:`str`, `optional`): Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.
        from_tf (:obj:`bool`, `optional`, defaults to `False`): Load the model weights from a TensorFlow checkpoint saved with the `save_pretrained()` method.
        *model_args (Tuple): All remaining positional arguments will be passed to the underlying model's ``__init__`` method.
        **model_kwargs (Dict): All remaining keyword arguments will be passed to the underlying model's ``__init__`` method.

    Returns:
        :class:`~transformers.PreTrainedModel`: An instance of a pretrained model.
    '''
    # Load the pretrained model
    model = DistilBertModel(model_name_or_path, *model_args, **kwargs)
    return model
