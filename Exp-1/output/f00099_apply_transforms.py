from typing import *
from datasets import Dataset

def apply_transforms(dataset, transforms):
    '''
    Apply transforms to a dataset.

    Args:
        dataset (Dataset): The dataset to apply transforms to.
        transforms (dict): The transforms to apply.

    Returns:
        None
    '''
    dataset.set_transform(transforms)
