# function_import --------------------

import numpy as np
from asteroid.models import ConvTasNet

# function_code --------------------

def separate_speaker_voices(wavs):
    """
    Separate speaker voices from mixed sound using ConvTasNet_Libri3Mix_sepclean_8k model.

    Args:
        wavs (numpy.ndarray): Array of mixed audio signals.

    Returns:
        numpy.ndarray: Array of separated audio signals.
    """
    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri3Mix_sepclean_8k")
    separated_audio = model.separate(wavs)
    return separated_audio

# test_function_code --------------------

def test_separate_speaker_voices():
    """
    Test separate_speaker_voices function.
    """
    # Test case: Single audio file
    wavs = np.random.rand(8000)
    separated_audio = separate_speaker_voices(wavs)
    assert isinstance(separated_audio, np.ndarray), 'Output should be a numpy array'

    # Test case: Multiple audio files
    wavs = np.random.rand(3, 8000)
    separated_audio = separate_speaker_voices(wavs)
    assert isinstance(separated_audio, np.ndarray), 'Output should be a numpy array'

    # Test case: Empty input
    wavs = np.array([])
    separated_audio = separate_speaker_voices(wavs)
    assert isinstance(separated_audio, np.ndarray), 'Output should be a numpy array'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_separate_speaker_voices()