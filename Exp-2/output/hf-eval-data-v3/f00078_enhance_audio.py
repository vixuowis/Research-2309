# function_import --------------------

import os
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audio(input_audio_file: str, output_audio_file: str) -> None:
    """
    Enhance the audio of noisy recordings using a pretrained Sepformer model.

    Args:
        input_audio_file (str): Path to the input audio file.
        output_audio_file (str): Path to save the enhanced audio file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input_audio_file does not exist.
    """
    model = separator.from_hparams(source='speechbrain/sepformer-whamr-enhancement', savedir='pretrained_models/sepformer-whamr-enhancement')
    est_sources = model.separate_file(path=input_audio_file)
    torchaudio.save(output_audio_file, est_sources[:, :, 0].detach().cpu(), 8000)

# test_function_code --------------------

def test_enhance_audio():
    """Tests for the `enhance_audio` function"""
    # Test case: Enhancing a sample audio file
    enhance_audio('sample_audio.wav', 'enhanced_audio.wav')
    assert os.path.exists('enhanced_audio.wav'), 'Test Failed: No output file created'
    
    # Test case: Input file does not exist
    try:
        enhance_audio('non_existent_file.wav', 'output.wav')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Test Failed: Expected a FileNotFoundError'
    
    print('All Tests Passed')

# call_test_function_code --------------------

test_enhance_audio()