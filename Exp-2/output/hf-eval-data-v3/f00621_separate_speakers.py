# function_import --------------------

import soundfile as sf
from asteroid import ConvTasNet
from huggingface_hub import hf_hub_download
import numpy as np

# function_code --------------------

def separate_speakers(audio_file):
    """
    This function separates speakers from a recorded audio using the ConvTasNet_Libri2Mix_sepclean_8k model from Hugging Face Transformers.

    Args:
        audio_file (str): The path to the audio file to be processed.

    Returns:
        numpy.ndarray: A 2D numpy array where each row corresponds to a separated speaker.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    model_weights = hf_hub_download(repo_id='JorisCos/ConvTasNet_Libri2Mix_sepclean_8k', filename='model.pth')
    model = ConvTasNet.from_pretrained(model_weights)
    mixture_audio, sample_rate = sf.read(audio_file)
    est_sources = model.separate(mixture_audio)
    return est_sources

# test_function_code --------------------

def test_separate_speakers():
    """
    This function tests the `separate_speakers` function with a sample audio file.
    """
    # Test with a sample audio file
    est_sources = separate_speakers('sample_audio.wav')
    assert isinstance(est_sources, np.ndarray), 'The output should be a numpy array.'
    assert est_sources.ndim == 2, 'The output should be a 2D array.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_separate_speakers()