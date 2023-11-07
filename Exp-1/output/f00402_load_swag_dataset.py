from typing import *
from datasets import load_dataset

def load_swag_dataset():
    """
    Load SWAG dataset

    Start by loading the `regular` configuration of the SWAG dataset from the 🤗 Datasets library:
    """
    swag = load_dataset("swag", "regular")
