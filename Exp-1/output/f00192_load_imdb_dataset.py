from typing import *
from datasets import load_dataset

def load_imdb_dataset():
    """Load IMDb dataset

    Start by loading the IMDb dataset from the 🤗 Datasets library:

    Returns:
        imdb: The loaded IMDb dataset
    """
    imdb = load_dataset("imdb")
    return imdb
