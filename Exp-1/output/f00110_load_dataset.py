from typing import *
from datasets import load_dataset

def load_dataset(dataset_name: str, **kwargs) -> DatasetDict:
    """Loads a dataset from the datasets library."""
    return datasets.load_dataset(dataset_name, **kwargs)
