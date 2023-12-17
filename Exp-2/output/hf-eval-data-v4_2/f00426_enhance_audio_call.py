# requirements_file --------------------

!pip install -U torch torchaudio speechbrain

# function_import --------------------

import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement

# function_code --------------------

def enhance_audio_call(input_audio_path, output_audio_path):
    """
    Enhances a noisy audio call using a pre-trained MetricGAN model.

    Args:
        input_audio_path (str): The file path to the noisy audio file.
        output_audio_path (str): The file path where the enhanced audio will be saved.

    Returns:
        str: The file path of the enhanced audio.

    Raises:
        FileNotFoundError: If the input audio file does not exist.
        Exception: If enhancement fails.
    """
    # Load the pre-trained MetricGAN model
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source='speechbrain/metricgan-plus-voicebank',
        savedir='pretrained_models/metricgan-plus-voicebank',
    )

    # Check if the input audio file exists
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError('The input audio file does not exist.')

    # Load the noisy audio file
    noisy = enhance_model.load_audio(input_audio_path).unsqueeze(0)

    # Enhance the audio
    enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))

    # Save the enhanced audio
    torchaudio.save(output_audio_path, enhanced.cpu(), 16000)

    # Return the path to the enhanced audio
    return output_audio_path

# test_function_code --------------------

def test_enhance_audio_call():
    print("Testing started.")

    # Test case 1: Check if the function enhances an audio file correctly
    print("Testing case [1/1] started.")
    enhanced_audio_path = enhance_audio_call('coworker_call_noisy.wav', 'coworker_call_enhanced.wav')
    assert os.path.exists(enhanced_audio_path), f"Test case [1/1] failed: Enhanced audio file not found at path {enhanced_audio_path}."
    print("Testing finished.")

# call_test_function_line --------------------

test_enhance_audio_call()