from typing import *
from datasets import Dataset

def train_test_split(self, test_size: Union[float, int] = 0.1, shuffle: bool = True, seed: Optional[int] = None) -> Tuple['Dataset', 'Dataset']:
    """Splits the dataset into a train and test set.

    This method splits the dataset's `train` split into a train and test set.

    Args:
        test_size (float or int, optional): The proportion or absolute number of samples to include in the test set.
            If float, it represents the proportion of the dataset to include in the test set.
            If int, it represents the absolute number of samples to include in the test set.
            Defaults to 0.1.
        shuffle (bool, optional): Whether or not to shuffle the dataset before splitting.
            Defaults to True.
        seed (int, optional): Random seed to use for shuffling the dataset.

    Returns:
        Tuple[Dataset, Dataset]: A tuple containing the train and test sets.
    """
    train, test = self._train_test_split(test_size, shuffle, seed)

    return Dataset.from_dict(train), Dataset.from_dict(test)
