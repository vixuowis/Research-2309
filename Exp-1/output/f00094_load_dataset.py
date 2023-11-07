from typing import *
from datasets import load_dataset

def load_dataset(dataset_name: str, split: Optional[Union[str, Split]] = None, **kwargs) -> DatasetDict:
    """Loads a dataset from the 🤗 Datasets library.

    This function is just a convenient way to directly access the datasets hosted on the hub
    without having to go through the datasets library, that can be a bit verbose. This
    is the recommended way to use datasets in scripts or notebooks (e.g., instead of
    using `datasets.load_dataset`)

    Args:
        dataset_name (str): The name of the dataset to load.
        split (:obj:`str` or `datasets.Split`): Which split of the data to load.
            If not provided, will return all splits in a `datasets.DatasetDict`.
        **kwargs: Keyword arguments that will be passed to the dataset loading script.

    Returns:
        `datasets.DatasetDict`: The dataset, or if `split` is specified, the split requested.
    """
    # Implementation details
    pass
