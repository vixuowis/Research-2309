from typing import *
from typing import List, Dict

def filter_long_documents(dataset: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Filters out long documents from the dataset.
    
    Args:
    - dataset (List[Dict[str, str]]): The input dataset containing documents.
    
    Returns:
    - filtered_dataset (List[Dict[str, str]]): The filtered dataset with long documents removed.
    """
    filtered_dataset = []
    for document in dataset:
        if len(document['words']) + len(document['question'].split()) < 512:
            filtered_dataset.append(document)
    return filtered_dataset
