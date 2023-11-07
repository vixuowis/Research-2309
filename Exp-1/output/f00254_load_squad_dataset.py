from typing import *
from datasets import load_dataset

def load_squad_dataset():
    """Load a smaller subset of the SQuAD dataset from the 🤗 Datasets library.

    Returns:
        dataset: The loaded dataset.
    """
    dataset = load_dataset("squad", split="train[:5000]")
    return dataset
