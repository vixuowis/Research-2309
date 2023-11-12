# function_import --------------------

import os
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audio_quality(input_audio_path: str, output_audio_path: str = 'enhanced_audio.wav') -> str:
    """
    Enhance the quality of an audio file using a pre-trained model from SpeechBrain.

    Args:
        input_audio_path (str): Path to the input low-quality audio file.
        output_audio_path (str, optional): Path to save the enhanced audio file. Defaults to 'enhanced_audio.wav'.

    Returns:
        str: Path to the enhanced audio file.

    Raises:
        FileNotFoundError: If the input audio file does not exist.
        RuntimeError: If there is an error during the enhancement process.
    """
    if not os.path.exists(input_audio_path):
        raise FileNotFoundError(f'Input audio file {input_audio_path} does not exist.')

    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    est_sources = model.separate_file(path=input_audio_path)
    torchaudio.save(output_audio_path, est_sources[:, :, 0].detach().cpu(), 16000)

    return output_audio_path

# test_function_code --------------------

def test_enhance_audio_quality():
    """Test the enhance_audio_quality function."""
    # Test with a valid audio file
    input_audio_path = 'path_to_low_quality_audio.wav'
    output_audio_path = 'enhanced_audio.wav'
    assert os.path.exists(enhance_audio_quality(input_audio_path, output_audio_path))

    # Test with a non-existent audio file
    try:
        enhance_audio_quality('non_existent_file.wav')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Expected a FileNotFoundError.'

    # Test with a valid audio file but invalid output path
    try:
        enhance_audio_quality(input_audio_path, '/invalid/path/enhanced_audio.wav')
    except RuntimeError:
        pass
    else:
        assert False, 'Expected a RuntimeError.'

    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_enhance_audio_quality())