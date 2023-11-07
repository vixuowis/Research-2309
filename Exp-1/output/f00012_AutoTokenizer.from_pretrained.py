from typing import *
from transformers import AutoTokenizer

def from_pretrained(model_name_or_path: Union[str, os.PathLike], *inputs, **kwargs) -> PreTrainedTokenizer:
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
