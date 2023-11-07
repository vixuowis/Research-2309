from typing import *
from transformers import TFPreTrainedModel

def prepare_tf_dataset(self, tokenized_data, shuffle, batch_size, collate_fn):
    """Converts the tokenized data to the `tf.data.Dataset` format.

    Args:
        tokenized_data (List[Dict[str, np.ndarray]]): The tokenized data.
        shuffle (bool): Whether to shuffle the data.
        batch_size (int): The batch size.
        collate_fn (Callable): The collate function.

    Returns:
        tf.data.Dataset: The converted dataset.
    """
    dataset = tf.data.Dataset.from_generator(
        lambda: tokenized_data,
        output_signature={
            'input_ids': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'attention_mask': tf.TensorSpec(shape=(None,), dtype=tf.int32),
            'labels': tf.TensorSpec(shape=(None,), dtype=tf.int32),
        }
    )

    dataset = dataset.shuffle(buffer_size=len(tokenized_data)) if shuffle else dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(collate_fn)

    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
