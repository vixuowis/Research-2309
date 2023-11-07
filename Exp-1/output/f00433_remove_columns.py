from typing import *
from datasets import Dataset

def remove_columns(dataset, columns):
    """Remove specified columns from the dataset.

    Args:
        dataset (Dataset): The input dataset.
        columns (Union[str, List[str]]): The name(s) of the column(s) to remove.

    Returns:
        Dataset: The modified dataset with the specified columns removed."""
    return dataset.remove_columns(columns)
