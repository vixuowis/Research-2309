# function_import --------------------

import torch
from transformers import Text2Speech

# function_code --------------------

def convert_text_to_speech(text):
    """
    Convert the given text to speech using a pretrained model from ESPnet.

    Args:
        text (str): The text to be converted to speech.

    Returns:
        Tensor: The synthesized speech output.
    """
    model = Text2Speech.from_pretrained('espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan')
    speech_output = model(text)
    return speech_output

# test_function_code --------------------

def test_convert_text_to_speech():
    """
    Test the convert_text_to_speech function.
    """
    text = 'Hello, world!'
    speech_output = convert_text_to_speech(text)
    assert isinstance(speech_output, torch.Tensor), 'The output should be a Tensor.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_convert_text_to_speech()