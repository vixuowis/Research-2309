# function_import --------------------

from transformers import AutoModelForAudioToAudio

# function_code --------------------

def apply_noise_suppression(noisy_audio_input):
    """
    This function applies noise suppression to an audio input using a pre-trained DCCRNet model.

    Args:
        noisy_audio_input (AudioData): The noisy audio input that needs noise suppression.

    Returns:
        AudioData: The denoised audio output.

    Raises:
        ValueError: If the input is not a valid AudioData instance.
    """
    model = AutoModelForAudioToAudio.from_pretrained('JorisCos/DCCRNet_Libri1Mix_enhsingle_16k')
    denoised_audio = model(noisy_audio_input)
    return denoised_audio

# test_function_code --------------------

def test_apply_noise_suppression():
    """
    This function tests the apply_noise_suppression function by using a sample noisy audio input and checking if the output is of the correct type.
    """
    noisy_audio_input = load_sample_noisy_audio()
    denoised_audio = apply_noise_suppression(noisy_audio_input)
    assert isinstance(denoised_audio, type(noisy_audio_input)), 'The output should be of the same type as the input.'

# call_test_function_code --------------------

test_apply_noise_suppression()