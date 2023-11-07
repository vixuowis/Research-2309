from typing import *
from transformers import AutoModelForTokenClassification


def from_pretrained(model_name_or_path: Union[str, os.PathLike], *model_args, **kwargs) -> PreTrainedModel:
    
    # Load a pretrained model either from a local directory or from a remote repository.
    def from_pretrained(model_name_or_path: Union[str, os.PathLike], *model_args, **kwargs) -> PreTrainedModel:
        
        # Load the model architecture from the specified path or repository
        model = AutoModelForTokenClassification.from_pretrained(model_name_or_path, *model_args, **kwargs)
        
        return model
