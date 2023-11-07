from typing import *
from typing import List

from datasets import Dataset

def preprocess_function(example: dict) -> dict:
    # Preprocess the example
    preprocessed_example = {}
    preprocessed_example['input'] = example['input'].lower()
    preprocessed_example['output'] = example['output'].upper()

    # Return the preprocessed example
    return preprocessed_example
