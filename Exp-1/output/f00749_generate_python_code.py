from typing import *
from transformers import HfAgent

def generate_python_code(prompt: str, return_code: bool) -> str:
    """Generate Python code based on the given prompt.

    Args:
        prompt (str): The prompt to generate code from.
        return_code (bool): Whether to return the generated code or not.

    Returns:
        str: The generated Python code if return_code is True, otherwise an empty string.
    """
    agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder")

    if return_code:
        return agent.run(prompt, return_code=True)
    else:
        agent.run(prompt)
