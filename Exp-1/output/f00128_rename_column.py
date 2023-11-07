from typing import *
def rename_column(column_name, new_name):
    """Renames a column in the dataset.

    Args:
        column_name (str): The name of the column to be renamed.
        new_name (str): The new name for the column.

    Returns:
        Dataset: The dataset with the renamed column."""
    return self.map(lambda example: {new_name: example[column_name] if column_name != new_name else example[column_name] for column_name in example})
