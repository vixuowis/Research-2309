# function_import --------------------

import torchaudio
from speechbrain.pretrained import Tacotron2, HIFIGAN

# function_code --------------------

def text_to_speech(text: str, output_file: str = 'example_TTS.wav'):
    """
    Converts the given text into speech and saves the output as a .wav file.

    Args:
        text (str): The text to be converted into speech.
        output_file (str, optional): The name of the output .wav file. Defaults to 'example_TTS.wav'.

    Returns:
        None
    """
    tacotron2 = Tacotron2.from_hparams(source='speechbrain/tts-tacotron2-ljspeech')
    hifi_gan = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-ljspeech')
    mel_output, mel_length, alignment = tacotron2.encode_text(text)
    waveforms = hifi_gan.decode_batch(mel_output)
    torchaudio.save(output_file, waveforms.squeeze(1), 22050)

# test_function_code --------------------

def test_text_to_speech():
    """
    Tests the text_to_speech function by converting a sample text into speech and checking if the output file is created.
    """
    import os
    sample_text = 'The sun was shining brightly, and the birds were singing sweetly.'
    output_file = 'test_TTS.wav'
    text_to_speech(sample_text, output_file)
    assert os.path.exists(output_file), 'Output file not created.'
    os.remove(output_file)

# call_test_function_code --------------------

test_text_to_speech()