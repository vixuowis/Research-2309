from typing import *
from typing import List

def convert_ner_tags(ner_tags: List[int]) -> List[str]:
    """Converts the numbers in ner_tags to their corresponding label names.

    Args:
        ner_tags (List[int]): The list of numbers representing entities.

    Returns:
        List[str]: The list of label names corresponding to the numbers in ner_tags.
    """
    label_list = [
        "O",
        "B-corporation",
        "I-corporation",
        "B-creative-work",
        "I-creative-work",
        "B-group",
        "I-group",
        "B-location",
        "I-location",
        "B-person",
        "I-person",
        "B-product",
        "I-product",
    ]

    label_names = [label_list[tag] for tag in ner_tags]

    return label_names
