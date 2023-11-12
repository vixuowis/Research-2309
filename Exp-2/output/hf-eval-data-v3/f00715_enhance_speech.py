# function_import --------------------

import os
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_speech(input_audio_path: str, output_audio_path: str) -> None:
    """
    Enhance the speech in an audio file using the SepFormer model from SpeechBrain.

    Args:
        input_audio_path (str): Path to the input audio file.
        output_audio_path (str): Path to save the enhanced audio file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input audio file does not exist.
        Exception: If there is an error in enhancing the speech in the audio.
    """
    try:
        # Load the pre-trained SepFormer model
        model = separator.from_hparams(source='speechbrain/sepformer-wham-enhancement', savedir='pretrained_models/sepformer-wham-enhancement')
        # Enhance the speech in the audio file
        est_sources = model.separate_file(path=input_audio_path)
        # Save the enhanced audio file
        torchaudio.save(output_audio_path, est_sources[:, :, 0].detach().cpu(), 8000)
    except FileNotFoundError as fnf_error:
        print(f'Error: {fnf_error}')
    except Exception as e:
        print(f'Error: {e}')

# test_function_code --------------------

def test_enhance_speech():
    """
    Test the enhance_speech function.
    """
    # Test case 1: Enhance the speech in a normal audio file
    enhance_speech('normal_audio.wav', 'enhanced_normal_audio.wav')
    assert os.path.exists('enhanced_normal_audio.wav')

    # Test case 2: Enhance the speech in a noisy audio file
    enhance_speech('noisy_audio.wav', 'enhanced_noisy_audio.wav')
    assert os.path.exists('enhanced_noisy_audio.wav')

    # Test case 3: Enhance the speech in a low volume audio file
    enhance_speech('low_volume_audio.wav', 'enhanced_low_volume_audio.wav')
    assert os.path.exists('enhanced_low_volume_audio.wav')

    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_enhance_speech())