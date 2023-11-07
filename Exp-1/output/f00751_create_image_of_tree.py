from typing import *
from agent import Agent

def create_image_of_tree(agent: Agent) -> int:
    """Create an image of a tree

    Args:
        agent (Agent): The agent object

    Returns:
        int: The return code
    """
    return agent.run("Create an image of a tree", return_code=True)
