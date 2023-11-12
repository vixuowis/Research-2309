# function_import --------------------

import librosa
import numpy as np
from asteroid.models import ConvTasNet_Libri3Mix_sepclean_8k

# function_code --------------------

def separate_speaker_sources(audio_file_path):
    """
    Separate the speaker sources from the original audio file.

    Args:
        audio_file_path (str): Path to the audio file.

    Returns:
        numpy.ndarray: Separated sources from the audio file.
    """
    model = ConvTasNet_Libri3Mix_sepclean_8k()
    audio, _ = librosa.load(audio_file_path, sr=None, mono=False)
    sep_sources = model.separate(audio)
    return sep_sources

# test_function_code --------------------

def test_separate_speaker_sources():
    """
    Test the function separate_speaker_sources.
    """
    # Test with a sample audio file
    sep_sources = separate_speaker_sources('sample_audio.wav')
    assert isinstance(sep_sources, np.ndarray), 'The returned object is not a numpy array.'
    assert sep_sources.shape[0] > 0, 'The returned array is empty.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_separate_speaker_sources()