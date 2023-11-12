# function_import --------------------

import os
from speechbrain.pretrained import Tacotron2, HIFIGAN
import torchaudio

# function_code --------------------

def text_to_speech(text: str, output_file: str = 'example_TTS.wav'):
    """
    Convert the provided text into speech and save it as a .wav file.

    Args:
        text (str): The text to be converted into speech.
        output_file (str): The name of the output .wav file. Default is 'example_TTS.wav'.

    Returns:
        None

    Raises:
        ModuleNotFoundError: If the required modules are not installed.
    """
    tacotron2 = Tacotron2.from_hparams(source='speechbrain/tts-tacotron2-ljspeech')
    hifi_gan = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech')
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)
    torchaudio.save(output_file, waveforms.squeeze(1), 22050)

# test_function_code --------------------

def test_text_to_speech():
    """
    Test the text_to_speech function with different inputs.
    """
    text_to_speech('Hello world')
    assert os.path.exists('example_TTS.wav'), 'File not found'
    os.remove('example_TTS.wav')
    text_to_speech('This is a test', 'test_TTS.wav')
    assert os.path.exists('test_TTS.wav'), 'File not found'
    os.remove('test_TTS.wav')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_text_to_speech()