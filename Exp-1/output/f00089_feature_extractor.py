from typing import *
from typing import List, Dict
import numpy as np

def feature_extractor(audio: List[np.ndarray], sampling_rate: int) -> Dict[str, List[np.ndarray]]:
    """Pass the audio `array` to the feature extractor. We also recommend adding the `sampling_rate` argument in the feature extractor in order to better debug any silent errors that may occur."""
    input_values = [np.array(audio[0], dtype=np.float32)]
    return {'input_values': input_values}
