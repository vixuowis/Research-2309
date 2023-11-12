# function_import --------------------

import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement
import os

# function_code --------------------

def enhance_audio(input_file: str, output_file: str):
    """
    Enhance the quality of an audio file using a pre-trained model.

    Args:
        input_file (str): Path to the input audio file.
        output_file (str): Path to save the enhanced audio file.

    Returns:
        None
    """
    enhance_model = SpectralMaskEnhancement.from_hparams(
        source='speechbrain/metricgan-plus-voicebank',
        savedir='pretrained_models/metricgan-plus-voicebank',
    )
    noisy = enhance_model.load_audio(input_file).unsqueeze(0)
    enhanced = enhance_model.enhance_batch(noisy, lengths=torch.tensor([1.]))
    torchaudio.save(output_file, enhanced.cpu(), 16000)

# test_function_code --------------------

def test_enhance_audio():
    """
    Test the enhance_audio function.
    """
    # Test with a sample audio file
    enhance_audio('sample_noisy.wav', 'sample_enhanced.wav')
    assert os.path.exists('sample_enhanced.wav'), 'Enhanced audio file not found.'
    # Clean up
    os.remove('sample_enhanced.wav')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_enhance_audio()