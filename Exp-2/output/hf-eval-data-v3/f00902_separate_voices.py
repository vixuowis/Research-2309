# function_import --------------------

import numpy as np
from huggingface_hub import hf_hub_download
from asteroid import ConvTasNet
from asteroid.utils.hub_utils import load_model

# function_code --------------------

def separate_voices(audio):
    '''
    Separate the voices in a single channel audio recording using a pre-trained model from Hugging Face.

    Args:
        audio (numpy array): The single channel audio recording.

    Returns:
        numpy array: The separated voices of the two speakers.
    '''
    repo_id = 'JorisCos/ConvTasNet_Libri2Mix_sepclean_8k'
    filename = hf_hub_download(repo_id, 'model.pth')
    model = load_model(filename)
    separated_sources = model(audio)
    return separated_sources

# test_function_code --------------------

def test_separate_voices():
    '''
    Test the separate_voices function.
    '''
    # Load a test audio file
    audio = np.random.rand(8000)
    separated_sources = separate_voices(audio)
    assert separated_sources.shape == audio.shape, 'The shape of the separated sources should be the same as the input audio.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_separate_voices()