from typing import *
from hf_agent import HfAgent

def instantiate_agent(controlnet_transformer, upscaler):
    """Instantiate an agent with controlnet_transformer and upscaler.

    Args:
        controlnet_transformer (str): The controlnet_transformer tool.
        upscaler (str): The upscaler tool.

    Returns:
        HfAgent: The instantiated agent.
    """
    tools = [controlnet_transformer, upscaler]
    agent = HfAgent("https://api-inference.huggingface.co/models/bigcode/starcoder", additional_tools=tools)
    return agent
