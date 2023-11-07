from typing import *
from typing import Dict

def get_audio_path(data: Dict[str, any]) -> str:
    '''
    Returns the path of the audio file in the given data dictionary.

    Params:
    - data (Dict[str, any]): The data dictionary.

    Returns:
    - str: The path of the audio file.
    '''
    return data['audio']['path']
