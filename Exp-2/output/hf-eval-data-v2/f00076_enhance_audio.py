# function_import --------------------

import torch
import torchaudio
from speechbrain.pretrained import SpectralMaskEnhancement

# function_code --------------------

def enhance_audio(input_audio_path: str, output_audio_path: str = 'enhanced.wav'):
    """
    Enhance the quality of an audio file by removing noise.

    Args:
        input_audio_path (str): Path to the input audio file.
        output_audio_path (str, optional): Path to save the enhanced audio file. Defaults to 'enhanced.wav'.

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

    Raises:
        AssertionError: If the function does not work as expected.
    """
    # Use a sample audio file for testing
    input_audio_path = 'path/to/sample_audio.wav'
    output_audio_path = 'path/to/enhanced_audio.wav'
    enhance_audio(input_audio_path, output_audio_path)
    # Load the enhanced audio file
    enhanced_audio, _ = torchaudio.load(output_audio_path)
    # Check if the audio file has been enhanced
    assert enhanced_audio.shape[0] == 1, 'The enhanced audio should be mono.'

# call_test_function_code --------------------

test_enhance_audio()