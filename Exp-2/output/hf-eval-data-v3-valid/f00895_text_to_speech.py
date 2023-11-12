# function_import --------------------

import os
import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN

# function_code --------------------

def text_to_speech(text: str, output_file: str):
    '''
    Convert the input text into speech and save the audio to a .wav file.

    Args:
        text (str): The input text to be converted into speech.
        output_file (str): The path of the output .wav file.

    Returns:
        None
    '''
    tacotron2 = Tacotron2.from_hparams(source='speechbrain/tts-tacotron2-ljspeech')
    hifi_gan = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech')
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)
    torchaudio.save(output_file, waveforms.squeeze(1), 22050)

# test_function_code --------------------

def test_text_to_speech():
    '''
    Test the text_to_speech function.
    '''
    text_to_speech('The sun was shining brightly, and the birds were singing sweetly.', 'test_TTS.wav')
    assert os.path.exists('test_TTS.wav'), 'The output file does not exist.'
    os.remove('test_TTS.wav')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_text_to_speech()