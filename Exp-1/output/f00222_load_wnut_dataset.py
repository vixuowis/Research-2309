from typing import *
from datasets import load_dataset

def load_wnut_dataset():
    """Load the WNUT 17 dataset from the 🤗 Datasets library.

    Returns:
        dataset: The loaded WNUT 17 dataset.
    """
    dataset = load_dataset("wnut_17")
    return dataset
