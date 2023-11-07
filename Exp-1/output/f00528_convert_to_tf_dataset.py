from typing import *
from transformers import DefaultDataCollator


def convert_to_tf_dataset(dataset, columns, shuffle, batch_size):
    data_collator = DefaultDataCollator(return_tensors='tf')
    tf_dataset = dataset.to_tf_dataset(
        columns=columns,
        shuffle=shuffle,
        batch_size=batch_size,
        collate_fn=data_collator
    )
    return tf_dataset
