# function_import --------------------

import os
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audio(input_audio_file: str, output_audio_file: str) -> None:
    """
    Enhance the quality of an audio file by removing noise.

    Args:
        input_audio_file (str): Path to the input audio file.
        output_audio_file (str): Path to save the enhanced audio file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input_audio_file does not exist.
    """
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    est_sources = model.separate_file(path=input_audio_file)
    torchaudio.save(output_audio_file, est_sources[:, :, 0].detach().cpu(), 16000)

# test_function_code --------------------

def test_enhance_audio():
    """
    Test the enhance_audio function.
    """
    # Test with a valid audio file
    enhance_audio('example_wham16k.wav', 'enhanced_wham16k.wav')
    assert os.path.exists('enhanced_wham16k.wav')

    # Test with a non-existing audio file
    try:
        enhance_audio('non_existing_file.wav', 'enhanced_non_existing_file.wav')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Expected a FileNotFoundError.')

    return 'All Tests Passed'

# call_test_function_code --------------------

test_enhance_audio()