from typing import *
from datasets import load_dataset, Audio

def load_minds14_dataset() -> Dataset:
    """
    Load the MInDS-14 dataset from the 🤗 Datasets library

    Returns:
        Dataset: The loaded MInDS-14 dataset
    """
    minds = load_dataset("PolyAI/minds14", name="en-US", split="train")

    return minds
