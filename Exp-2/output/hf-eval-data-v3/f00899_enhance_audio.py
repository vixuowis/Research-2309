# function_import --------------------

import os
import torchaudio
from speechbrain.pretrained import WaveformEnhancement

# function_code --------------------

def enhance_audio(input_audio_file: str, output_audio_file: str) -> None:
    """
    Enhance the quality of an audio file by reducing background noise.

    Args:
        input_audio_file (str): The path to the input audio file.
        output_audio_file (str): The path to the output audio file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the input audio file does not exist.
    """
    enhance_model = WaveformEnhancement.from_hparams(
        source="speechbrain/mtl-mimic-voicebank",
        savedir="pretrained_models/mtl-mimic-voicebank",
    )
    enhanced = enhance_model.enhance_file(input_audio_file)
    torchaudio.save(output_audio_file, enhanced.unsqueeze(0).cpu(), 16000)

# test_function_code --------------------

def test_enhance_audio():
    """
    Test the enhance_audio function.
    """
    # Test case: Enhance an audio file
    enhance_audio('test_audio.wav', 'enhanced_audio.wav')
    assert os.path.exists('enhanced_audio.wav'), 'The enhanced audio file was not created.'

    # Test case: Input audio file does not exist
    try:
        enhance_audio('non_existent_file.wav', 'enhanced_audio.wav')
    except FileNotFoundError:
        pass
    else:
        assert False, 'Expected a FileNotFoundError when the input audio file does not exist.'

    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_enhance_audio())