from typing import *
import os

def get_audio_path(data):
    """Get the audio path from the given data.

    Args:
        data (dict): The data containing the audio path.

    Returns:
        str: The audio path.
    """
    audio = data['audio']
    path = audio['path']
    return path
