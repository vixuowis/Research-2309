from typing import *
from transformers import AutoTokenizer

def get_tool_description(tool):
    """Get the description and name of a custom tool.

    Args:
        tool (str): The name of the custom tool.

    Returns:
        str: The description of the custom tool.
        str: The name of the custom tool.
    """
    if tool == 'controlnet_transformer':
        return controlnet_transformer.description, controlnet_transformer.name
    else:
        return '', ''
