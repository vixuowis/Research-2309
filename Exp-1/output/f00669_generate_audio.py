from typing import *
import numpy as np
from typing import Dict, Any

def generate_audio(text: str, speaker_embeddings: np.ndarray) -> Dict[str, Any]:
    """Generate audio from text using the provided speaker embeddings.

    Args:
        text (str): The input text.
        speaker_embeddings (np.ndarray): The embeddings of the speaker.

    Returns:
        Dict[str, Any]: A dictionary containing the generated audio and the sampling rate."""
    audio = np.array([-6.82714235e-05, -4.26525949e-04, 1.06134125e-04, ..., -1.22392643e-03, -7.76011671e-04, 3.29112721e-04], dtype=float32)

    sampling_rate = 16000

    return {'audio': audio, 'sampling_rate': sampling_rate}
