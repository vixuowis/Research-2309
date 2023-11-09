# function_import --------------------

import soundfile
from espnet2.bin.tts_inference import Text2Speech

# function_code --------------------

def convert_text_to_speech(text):
    """
    Convert Chinese text to speech using a pre-trained model from ESPnet.

    Args:
        text (str): The Chinese text to be converted to speech.

    Returns:
        None. The function writes the output speech to an audio file named 'out.wav'.
    """
    text2speech = Text2Speech.from_pretrained('espnet/kan-bayashi_csmsc_tts_train_tacotron2_raw_phn_pypinyin_g2p_phone_train.loss.best')
    speech = text2speech(text)["wav"]
    soundfile.write("out.wav", speech.numpy(), text2speech.fs)

# test_function_code --------------------

def test_convert_text_to_speech():
    """
    Test the convert_text_to_speech function.

    The function does not return a value. It writes the output speech to an audio file named 'out.wav'.
    Therefore, the test will pass if the function runs without errors.
    """
    text = '春江潮水连海平，海上明月共潮生'
    convert_text_to_speech(text)

# call_test_function_code --------------------

test_convert_text_to_speech()