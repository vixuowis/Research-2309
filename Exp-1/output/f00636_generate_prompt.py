from typing import *
def generate_prompt(question: str) -> str:
    """Generate a prompt for visual question answering task.

    Args:
        question (str): The question to be included in the prompt.

    Returns:
        str: The generated prompt.
    """
    prompt = f"Question: {question} Answer:"
    return prompt
