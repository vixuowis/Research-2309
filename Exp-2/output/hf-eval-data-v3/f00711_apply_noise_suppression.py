# function_import --------------------

import torch
from transformers import AutoModelForAudioToAudio

# function_code --------------------

def apply_noise_suppression(noisy_audio_input):
    """
    Apply noise suppression to voice commands using a pre-trained model.

    Args:
        noisy_audio_input (Tensor): The noisy audio input that needs noise suppression.

    Returns:
        Tensor: The denoised audio output.

    Raises:
        ImportError: If the required libraries are not installed.
    """
    model = AutoModelForAudioToAudio.from_pretrained('JorisCos/DCCRNet_Libri1Mix_enhsingle_16k')
    denoised_audio = model(noisy_audio_input)
    return denoised_audio

# test_function_code --------------------

def test_apply_noise_suppression():
    """
    Test the apply_noise_suppression function with a noisy audio input.
    """
    noisy_audio_input = torch.randn(1, 16000)
    denoised_audio = apply_noise_suppression(noisy_audio_input)
    assert denoised_audio.size() == noisy_audio_input.size(), 'The size of the denoised audio does not match the input size.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_apply_noise_suppression()