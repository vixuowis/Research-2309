from typing import *
from datasets import load_dataset, Audio

def load_minds_dataset():
    """
    Load a smaller subset of the MInDS-14 dataset from the 🤗 Datasets library.

    Returns:
        minds (datasets.Dataset): The loaded dataset.
    """
    minds = load_dataset("PolyAI/minds14", name="en-US", split="train[:100]")
    return minds
