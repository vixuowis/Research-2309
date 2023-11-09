# function_import --------------------

from huggingface_hub import hf_hub_download
from asteroid import ConvTasNet
from asteroid.utils.hub_utils import load_model

# function_code --------------------

def separate_voices(audio):
    '''
    This function separates the voices of two speakers in a single-channel audio recording.
    
    Args:
        audio (numpy array): The single-channel audio recording.
    
    Returns:
        numpy array: The separated voices of the two speakers.
    
    Raises:
        Exception: If the audio is not a single-channel recording.
    '''
    repo_id = 'JorisCos/ConvTasNet_Libri2Mix_sepclean_8k'
    filename = hf_hub_download(repo_id, 'model.pth')
    model = load_model(filename)
    separated_sources = model(audio)
    return separated_sources

# test_function_code --------------------

def test_separate_voices():
    '''
    This function tests the separate_voices function.
    
    Args:
        None
    
    Returns:
        None
    
    Raises:
        AssertionError: If the test fails.
    '''
    # Load a test audio recording
    test_audio = np.random.rand(8000)
    separated_sources = separate_voices(test_audio)
    assert separated_sources.shape == (2, 8000)

# call_test_function_code --------------------

test_separate_voices()