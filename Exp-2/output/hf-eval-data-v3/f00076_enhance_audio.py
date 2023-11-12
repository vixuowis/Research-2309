# function_import --------------------

import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement
import os

# function_code --------------------

def enhance_audio(input_audio_path: str, output_audio_path: str) -> None:
    """
    Enhance the quality of an audio file by removing noise.

    Args:
        input_audio_path (str): Path to the input noisy audio file.
        output_audio_path (str): Path to save the enhanced audio file.

    Returns:
        None
    """
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source='speechbrain/metricgan-plus-voicebank',
        savedir='pretrained_models/metricgan-plus-voicebank',
    )
    noisy = enhance_model.load_audio(input_audio_path).unsqueeze(0)
    enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))
    torchaudio.save(output_audio_path, enhanced.cpu(), 16000)

# test_function_code --------------------

def test_enhance_audio():
    """
    Test the enhance_audio function.
    """
    # Test case: Enhance a noisy audio file
    enhance_audio('path/to/noisy_audio_file.wav', 'path/to/enhanced_audio_file.wav')
    assert os.path.exists('path/to/enhanced_audio_file.wav'), 'Enhanced audio file not found.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_enhance_audio()