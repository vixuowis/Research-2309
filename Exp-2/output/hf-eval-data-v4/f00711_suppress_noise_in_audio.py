# requirements_file --------------------

!pip install -U transformers asteroid

# function_import --------------------

from transformers import AutoModelForAudioToAudio

# function_code --------------------

def suppress_noise_in_audio(noisy_audio_input):
    """
    This function applies noise suppression to an input audio signal using a pre-trained DCCRNet model.

    Args:
    noisy_audio_input : Tensor
        The raw audio signal tensor with noise.

    Returns:
    Tensor
        The denoised audio signal tensor.
    """
    # Load the pre-trained DCCRNet model
    model = AutoModelForAudioToAudio.from_pretrained('JorisCos/DCCRNet_Libri1Mix_enhsingle_16k')
    
    # Apply the model to the noisy audio input
    denoised_audio = model(noisy_audio_input)
    
    return denoised_audio

# test_function_code --------------------

def test_suppress_noise_in_audio():
    print("Testing started.")
    
    # This is a placeholder for loading a dataset with audio samples.
    # Ideally, we would load a real noisy audio sample to test.
    # For example, using torchaudio or librosa to load '.wav' files.
    dataset = load_dataset("librispeech_asr", "clean", split="validation")
    sample_data = dataset[0]['audio']['array']  # Get a sample audio array

    # Test case 1: Apply noise suppression to the audio sample
    print("Testing case [1/1] started.")
    denoised_audio = suppress_noise_in_audio(sample_data)
    # Ideally, we would assert the quality of the denoised audio.
    # However, for actual validation of denoising, one would need to compare audio metrics pre and post denoising.
    assert denoised_audio is not None, f"Test case [1/1] failed: The function did not return any output."
    
    print("Testing finished.")

# Run the test function (Note: This requires an actual noisy audio sample and appropriate dataset loading function)
test_suppress_noise_in_audio()