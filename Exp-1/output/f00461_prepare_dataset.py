from typing import *
from typing import Dict
from datasets import Dataset


def prepare_dataset(example: Dict[str, str]) -> Dict[str, str]:
    """
    Preprocesses a single example from the dataset.

    Args:
        example (Dict[str, str]): The example to preprocess.

    Returns:
        Dict[str, str]: The preprocessed example.
    """
    # Preprocess the example
    preprocessed_example = {}
    preprocessed_example['input'] = example['input'].strip().lower()
    preprocessed_example['output'] = example['output'].strip().lower()

    return preprocessed_example

