# requirements_file --------------------

!pip install -U transformers asteroid

# function_import --------------------

from transformers import AutoModelForAudioToAudio

# function_code --------------------

def suppress_noise_in_audio(noisy_audio_input):
    """Suppress noise in the provided audio input using a pre-trained deep learning model.

    Args:
        noisy_audio_input (np.ndarray): The noisy audio input array.

    Returns:
        np.ndarray: The denoised audio output.

    Raises:
        ValueError: If the input is not a valid audio array.
    """
    if not isinstance(noisy_audio_input, np.ndarray):
        raise ValueError('Input must be an audio array.')
    model = AutoModelForAudioToAudio.from_pretrained('JorisCos/DCCRNet_Libri1Mix_enhsingle_16k')
    denoised_audio = model(noisy_audio_input)
    return denoised_audio

# test_function_code --------------------

def test_suppress_noise_in_audio():
    print("Testing started.")
    # Simulate a noisy audio input for testing
    noisy_audio_input = np.random.randn(16000)  # 1 second of random noise at 16kHz

    # Testing case 1: Valid audio input
    print("Testing case [1/1] started.")
    denoised_audio = suppress_noise_in_audio(noisy_audio_input)
    assert denoised_audio is not None, "Test case [1/1] failed: No output from the function."
    print("Testing finished.")

# call_test_function_line --------------------

test_suppress_noise_in_audio()