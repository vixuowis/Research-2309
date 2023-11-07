from typing import *
from transformers import DistilBertForQuestionAnswering

def from_pretrained(model_name_or_path, config=None, *model_args, **kwargs):
    """Loads a pretrained model from a given model_name_or_path, config (if provided) and kwargs. The model is set in evaluation mode by default using model.eval() (Dropout modules are deactivated)."""
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased")
