# function_import --------------------

import os
from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

# function_code --------------------

def enhance_audio(input_file: str, output_file: str) -> None:
    """
    Enhance the quality of an audio file by removing background noise using a pre-trained model from SpeechBrain.

    Args:
        input_file (str): Path to the input audio file that needs speech enhancement.
        output_file (str): Path to save the enhanced audio file.

    Returns:
        None
    """
    model = separator.from_hparams(source='speechbrain/sepformer-wham16k-enhancement', savedir='pretrained_models/sepformer-wham16k-enhancement')
    est_sources = model.separate_file(path=input_file)
    torchaudio.save(output_file, est_sources[:, :, 0].detach().cpu(), 16000)

# test_function_code --------------------

def test_enhance_audio():
    """
    Test the enhance_audio function.
    """
    # Test case 1: Enhance a sample audio file
    enhance_audio('example_podcast.wav', 'enhanced_podcast.wav')
    assert os.path.exists('enhanced_podcast.wav')

    # Test case 2: Enhance another sample audio file
    enhance_audio('another_example_podcast.wav', 'another_enhanced_podcast.wav')
    assert os.path.exists('another_enhanced_podcast.wav')

    # Test case 3: Enhance a third sample audio file
    enhance_audio('third_example_podcast.wav', 'third_enhanced_podcast.wav')
    assert os.path.exists('third_enhanced_podcast.wav')

    return 'All Tests Passed'

# call_test_function_code --------------------

test_enhance_audio()