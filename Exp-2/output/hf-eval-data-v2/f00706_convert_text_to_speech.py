# function_import --------------------

import soundfile
from espnet2.bin.tts_inference import Text2Speech

# function_code --------------------

def convert_text_to_speech(lesson_text):
    """
    Convert the given Chinese text into an audio waveform using a pre-trained Text-to-Speech model.

    Args:
        lesson_text (str): The Chinese text to be converted into speech.

    Returns:
        A tuple (speech, fs), where speech is the audio waveform and fs is the sample rate.
    """
    text2speech = Text2Speech.from_pretrained('espnet/kan_bayashi_csmsc_tts_train_tacotron2_raw_phn_pypinyin_g2p_phone_train.loss.best')
    speech = text2speech(lesson_text)['wav']
    return speech.numpy(), text2speech.fs

# test_function_code --------------------

def test_convert_text_to_speech():
    """
    Test the convert_text_to_speech function with some example Chinese text.
    """
    lesson_text = '汉语很有趣'
    speech, fs = convert_text_to_speech(lesson_text)
    assert isinstance(speech, np.ndarray)
    assert isinstance(fs, int)
    soundfile.write('test_lesson_audio_example.wav', speech, fs, 'PCM_16')

# call_test_function_code --------------------

test_convert_text_to_speech()