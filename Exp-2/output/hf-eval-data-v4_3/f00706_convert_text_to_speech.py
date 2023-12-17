# requirements_file --------------------

import subprocess

requirements = ["soundfile", "torch", "espnet_model_zoo"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import soundfile
from espnet2.bin.tts_inference import Text2Speech

# function_code --------------------

def convert_text_to_speech(text):
    """
    Convert Chinese text to speech using a pre-trained Text-to-Speech model.

    Args:
        text (str): The Chinese text to be converted into speech.

    Returns:
        bytes: The audio data of the spoken text in WAV format.

    Raises:
        ValueError: If the input text is empty.
    """
    if not text:
        raise ValueError('Input text cannot be empty.')
    text2speech = Text2Speech.from_pretrained('espnet/kan-bayashi_csmsc_tts_train_tacotron2_raw_phn_pypinyin_g2p_phone_train.loss.best')
    speech = text2speech(text)['wav']
    audio_data = speech.numpy()
    return audio_data

# test_function_code --------------------

def test_convert_text_to_speech():
    print("Testing started.")

    # Testing case 1: Non-empty string
    print("Testing case [1/2] started.")
    assert convert_text_to_speech('汉语很有趣'), f"Test case [1/2] failed: Expected non-empty audio data."

    # Testing case 2: Empty string raises ValueError
    try:
        print("Testing case [2/2] started.")
        convert_text_to_speech('')
    except ValueError as e:
        assert str(e) == 'Input text cannot be empty.', f"Test case [2/2] failed: Expected ValueError with message 'Input text cannot be empty.'"

    print("Testing finished.")

# call_test_function_line --------------------

test_convert_text_to_speech()