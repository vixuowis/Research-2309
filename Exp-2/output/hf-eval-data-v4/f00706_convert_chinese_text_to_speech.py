# requirements_file --------------------

!pip install -U torch espnet_model_zoo soundfile

# function_import --------------------

import soundfile
from espnet2.bin.tts_inference import Text2Speech

# function_code --------------------

def convert_chinese_text_to_speech(text):
    """
    Convert Chinese text into an audio waveform using a pre-trained Text-to-Speech model.

    Parameters:
        text (str): The Chinese text to be converted.

    Returns:
        tuple: A tuple containing the audio waveform as a numpy array and the sample rate.
    """
    # Load the pre-trained TTS model
    text2speech = Text2Speech.from_pretrained('espnet/kan_bayashi_csmsc_tts_train_tacotron2_raw_phn_pypinyin_g2p_phone_train.loss.best')
    # Convert the Chinese text to the waveform
    speech = text2speech(text)['wav']

    # Return the waveform and the sample rate
    return speech.numpy(), text2speech.fs

# test_function_code --------------------

def test_convert_chinese_text_to_speech():
    print("Testing convert_chinese_text_to_speech function.")

    # Test case 1: Non-empty string
    print("Testing case [1/2] with non-empty string.")
    waveform, sample_rate = convert_chinese_text_to_speech("汉语很有趣")
    assert waveform is not None and waveform.size > 0, "Test case [1/2] failed: Returned waveform is empty."
    assert sample_rate > 0, "Test case [1/2] failed: Sample rate is not positive."

    # Test case 2: Empty string
    print("Testing case [2/2] with empty string.")
    waveform, sample_rate = convert_chinese_text_to_speech("")
    assert waveform is not None and waveform.size == 0, "Test case [2/2] failed: Returned waveform is not empty for empty input text."
    print("Testing finished.")

# Running the test function
test_convert_chinese_text_to_speech()