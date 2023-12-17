# requirements_file --------------------

!pip install -U numpy asteroid

# function_import --------------------

from asteroid.models import ConvTasNet

# function_code --------------------

def separate_speaker_voices(wavs):
    """
    Separate speaker voices from a mixed audio signal using the ConvTasNet model.

    Args:
        wavs (numpy.ndarray): An array of shape (batch, samples) containing the mixed audio signals to be separated.

    Returns:
        numpy.ndarray: An array of separated audio signals of shape (batch, n_sources, samples).

    Raises:
        ValueError: If the input is not a numpy array or if its shape is not compatible.
    """
    if not isinstance(wavs, np.ndarray) or len(wavs.shape) != 2:
        raise ValueError('Invalid input: expecting a numpy array of shape (batch, samples).')

    model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri3Mix_sepclean_8k")
    separated_audio = model.separate(wavs)
    return separated_audio


# test_function_code --------------------

def test_separate_speaker_voices():
    print("Testing started.")
    sample_mixed_audio = np.random.randn(1, 8000)  # Simulate a 1-second mixed audio signal with 8kHz sample rate

    # Test case 1: Check function with valid input
    print("Testing case [1/1] started.")
    separated = separate_speaker_voices(sample_mixed_audio)
    assert separated.shape == (1, 3, 8000), f"Test case [1/1] failed: expected shape (1, 3, 8000), got {separated.shape}"
    print("Testing finished.")


# call_test_function_line --------------------

test_separate_speaker_voices()