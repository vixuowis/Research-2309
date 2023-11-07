from typing import *
from transformers import TFPreTrainedModel

def prepare_tf_dataset(tokenized_books, shuffle, batch_size, collate_fn):
    """Convert datasets to tf.data.Dataset format

    Args:
        tokenized_books (dict): Tokenized datasets
        shuffle (bool): Whether to shuffle the dataset
        batch_size (int): Batch size
        collate_fn (Callable): Function to collate the data

    Returns:
        tf.data.Dataset: The converted dataset
    """
    tf_train_set = model.prepare_tf_dataset(
        tokenized_books["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
        tokenized_books["test"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )
