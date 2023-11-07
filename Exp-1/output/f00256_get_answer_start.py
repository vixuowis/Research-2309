from typing import *
from typing import List

def get_answer_start(answers: dict) -> List[int]:
    """Return the list of answer_start values from the answers dictionary.

    Args:
        answers (dict): A dictionary containing the answer_start values.

    Returns:
        List[int]: A list of answer_start values.
    """
    return answers['answer_start']
