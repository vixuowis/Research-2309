from typing import *
from typing import Any

def select_speaker(speaker_id: Any) -> bool:
    """
    This function filters the dataset based on the number of examples per speaker.

    Args:
    - speaker_id (Any): The ID of the speaker.

    Returns:
    - bool: True if the number of examples for the speaker is between 100 and 400, False otherwise.
    """
    return 100 <= speaker_counts[speaker_id] <= 400
