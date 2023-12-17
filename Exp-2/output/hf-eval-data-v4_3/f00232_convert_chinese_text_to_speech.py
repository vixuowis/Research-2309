# requirements_file --------------------

import subprocess

requirements = ["soundfile", "espnet"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

import soundfile
from espnet2.bin.tts_inference import Text2Speech

# function_code --------------------

def convert_chinese_text_to_speech(text: str) -> bytes:
    """Converts Chinese text to speech using ESPnet pre-trained model.

    Args:
        text (str): The Chinese text to be converted to speech.

    Returns:
        bytes: The speech data in byte format.

    Raises:
        ValueError: If the text is empty.
        RuntimeError: If the TTS model fails to generate speech.
    """
    if not text:
        raise ValueError('The input text is empty')

    # Load the pre-trained Chinese TTS model
    text2speech = Text2Speech.from_pretrained('espnet/kan-bayashi_csmsc_tts_train_tacotron2_raw_phn_pypinyin_g2p_phone_train.loss.best')

    # Convert text to speech
    speech = text2speech(text)["wav"]

    # Convert the numpy array wav data to bytes
    return speech.numpy().tobytes()

# test_function_code --------------------

def test_convert_chinese_text_to_speech():
    print("Testing started.")

    # Prepare a sample Chinese text
    sample_text = '春江潮水连海平，海上明月共潮生'

    # Test case 1: Normal input
    print("Testing case [1/1] started.")
    try:
        speech_data = convert_chinese_text_to_speech(sample_text)
        assert isinstance(speech_data, bytes), "Test case [1/1] failed: Speech data is not in bytes format."
    except Exception as e:
        assert False, f"Test case [1/1] failed: {e}"

    print("Testing finished.")

# call_test_function_line --------------------

test_convert_chinese_text_to_speech()