from typing import *
from transformers import AutoProcessor, AutoModelForQuestionAnswering

def generate_python_code(model_type: str) -> str:
    """
    Generates Python code based on the given model type.
    
    Args:
        model_type (str): The type of model to generate code for.
    
    Returns:
        str: The generated Python code.
    """
    if model_type not in ['Albert', 'Bart', 'Bert', 'BigBird', 'BigBirdPegasus', 'Bloom']:
        return 'Invalid model type'
    
    processor = AutoProcessor.from_pretrained(model_type)
    model = AutoModelForQuestionAnswering.from_pretrained(model_type)
    
    return f'processor = AutoProcessor.from_pretrained("{model_type}")\nmodel = AutoModelForQuestionAnswering.from_pretrained("{model_type}")'
