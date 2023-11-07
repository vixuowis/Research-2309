from typing import *
from typing import Dict

def id_to_label(label_id: str) -> str:
    """Converts a label ID to a label name.

    :param label_id: The ID of the label to convert.
    :return: The name of the label corresponding to the ID."""
    return id2label[str(label_id)]
