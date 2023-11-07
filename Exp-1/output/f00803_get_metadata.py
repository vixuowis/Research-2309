from typing import *
import json

def get_metadata(index):
    """
    Get the metadata of the index.

    Args:
        index (dict): The index dictionary.

    Returns:
        dict: The metadata dictionary.
    """
    metadata = index.get('metadata', {})
    return metadata
