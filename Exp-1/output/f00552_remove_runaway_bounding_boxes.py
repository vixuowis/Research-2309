from typing import *
from typing import List
from datasets import Dataset

def remove_runaway_bounding_boxes(dataset: Dataset, remove_idx: List[int]) -> Dataset:
    """
    Remove images with runaway bounding boxes from the dataset.

    Args:
        dataset (Dataset): The dataset to remove runaway bounding boxes from.
        remove_idx (List[int]): The indices of the images to remove.

    Returns:
        Dataset: The dataset with runaway bounding boxes removed.
    """
    keep = [i for i in range(len(dataset)) if i not in remove_idx]
    return dataset.select(keep)
