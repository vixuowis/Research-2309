# function_import --------------------

import os
import soundfile
from espnet2.bin.tts_inference import Text2Speech

# function_code --------------------

def convert_text_to_speech(text: str, model_name: str = 'espnet/kan-bayashi_csmsc_tts_train_tacotron2_raw_phn_pypinyin_g2p_phone_train.loss.best', output_file: str = 'out.wav'):
    """
    Convert the given Chinese text to speech using a pre-trained ESPnet model.

    Args:
        text (str): The Chinese text to be converted to speech.
        model_name (str, optional): The name of the pre-trained ESPnet model. Defaults to 'espnet/kan-bayashi_csmsc_tts_train_tacotron2_raw_phn_pypinyin_g2p_phone_train.loss.best'.
        output_file (str, optional): The name of the output file where the generated speech will be saved. Defaults to 'out.wav'.

    Returns:
        None

    Raises:
        ModuleNotFoundError: If the required libraries are not installed.
    """
    text2speech = Text2Speech.from_pretrained(model_name)
    speech = text2speech(text)["wav"]
    soundfile.write(output_file, speech.numpy(), text2speech.fs)

# test_function_code --------------------

def test_convert_text_to_speech():
    """
    Test the convert_text_to_speech function with some test cases.
    """
    # Test case 1: Normal case
    convert_text_to_speech('春江潮水连海平，海上明月共潮生')
    assert os.path.exists('out.wav'), 'Test case 1 failed'

    # Test case 2: Change output file name
    convert_text_to_speech('春江潮水连海平，海上明月共潮生', output_file='test.wav')
    assert os.path.exists('test.wav'), 'Test case 2 failed'

    # Test case 3: Use a different model
    convert_text_to_speech('春江潮水连海平，海上明月共潮生', model_name='espnet/other_model')
    assert os.path.exists('out.wav'), 'Test case 3 failed'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_convert_text_to_speech()