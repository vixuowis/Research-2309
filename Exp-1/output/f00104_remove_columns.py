from typing import *
import datasets

def remove_columns(dataset, columns):
    """Remove specified columns from the dataset.

    Args:
        dataset (datasets.Dataset): The dataset to remove columns from.
        columns (List[str]): The list of column names to remove.

    Returns:
        datasets.Dataset: The modified dataset with specified columns removed.
    """
    dataset = dataset.remove_columns(columns=columns)
    return dataset
