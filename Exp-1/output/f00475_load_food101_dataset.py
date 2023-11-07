from typing import *
from datasets import load_dataset

def load_food101_dataset():
    """Load a smaller subset of the Food-101 dataset from the 🤗 Datasets library.

    Returns:
        Dataset: The loaded dataset.
    """
    food = load_dataset("food101", split="train[:5000]")
    return food
