from typing import *
from datasets import Dataset

def remove_columns(dataset, columns):
    """Remove specified columns from the dataset.

    Args:
        dataset (Dataset): The dataset to remove columns from.
        columns (Union[str, List[str]]): The column(s) to remove.

    Returns:
        Dataset: The dataset with the specified columns removed.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        dataset = dataset.remove_column(column)

    return dataset
