from typing import *
from huggingface_hub import Audio

def cast_column(self, column: str, audio: Audio) -> Dataset:
    """Cast a column to a specific type.

    Args:
        column (str): The name of the column to cast.
        audio (Audio): The audio type to cast to.

    Returns:
        Dataset: The dataset with the column casted to the specified type."""
    return self.map(
        lambda example: {column: audio(example[column].numpy(), example[column].numpy().shape[0])},
        batched=True,
        remove_columns=[column],
    )
