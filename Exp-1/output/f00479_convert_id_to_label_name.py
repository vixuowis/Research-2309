from typing import *
from typing import Dict

def convert_id_to_label_name(id2label: Dict[str, str], label_id: str) -> str:
    """
    Converts the label id to a label name.

    Args:
        id2label (Dict[str, str]): A dictionary mapping label ids to label names.
        label_id (str): The label id to convert.

    Returns:
        str: The corresponding label name.
    """
    label_name = id2label.get(label_id)
    return label_name
