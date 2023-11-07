from typing import *
from transformers import AutoModelForSeq2SeqLM

def from_pretrained(model_name_or_path: str, low_cpu_mem_usage: bool = False, **kwargs) -> PreTrainedModel:
    
    Load a pretrained model from a given model identifier or path.

    Args:
        model_name_or_path (:obj:`str`): The identifier of the pre-trained model or the path to the directory containing the pre-trained model.
        low_cpu_mem_usage (:obj:`bool`, `optional`, defaults to :obj:`False`): Whether to activate low CPU memory usage mode.
        **kwargs: Additional keyword arguments passed along to the specific model's `from_pretrained` method.

    Returns:
        :class:`~transformers.PreTrainedModel`: The model instance.

t0pp = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0pp", low_cpu_mem_usage=True)
