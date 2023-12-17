# requirements_file --------------------

!pip install -U transformers torch espnet_model_zoo

# function_import --------------------

import torch
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech

# function_code --------------------

def convert_text_to_speech(text):
    """Convert a given text message to audio using a pretrained Text-to-Speech model.

    Args:
        text (str): The text message to be converted into speech.

    Returns:
        numpy.ndarray: The audio array of the synthesized speech.

    Raises:
        ValueError: If the text is empty or None.
    """
    if not text:
        raise ValueError('The text input cannot be empty.')

    model_downloader = ModelDownloader()
    text2speech = Text2Speech.from_pretrained('espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan')
    wav, _, _ = text2speech(text)
    return wav

# test_function_code --------------------

def test_convert_text_to_speech():
    print("Testing started.")
    sample_text = "Hello, world!"

    # Testing case 1: Non-empty string
    print("Testing case [1/2] started.")
    wav = convert_text_to_speech(sample_text)
    assert isinstance(wav, numpy.ndarray), "Test case [1/2] failed: The result should be a numpy ndarray."
    
    # Testing case 2: Empty string
    print("Testing case [2/2] started.")
    try:
        convert_text_to_speech("")
    except ValueError as e:
        assert str(e) == 'The text input cannot be empty.', "Test case [2/2] failed: ValueError not raised for empty text."
    print("Testing finished.")

# call_test_function_line --------------------

test_convert_text_to_speech()