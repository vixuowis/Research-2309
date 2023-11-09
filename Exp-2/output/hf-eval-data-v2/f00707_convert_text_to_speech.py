# function_import --------------------

import torch
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech

# function_code --------------------

def convert_text_to_speech(text):
    """
    Convert a given text into synthesized speech using a pretrained Text-to-Speech model.

    Args:
        text (str): The text message to be converted into speech.

    Returns:
        wav (torch.Tensor): The synthesized speech in the form of a waveform.

    Raises:
        Exception: If the text input is not a string.
    """
    if not isinstance(text, str):
        raise Exception('Input text must be a string.')
    d = ModelDownloader()
    text2speech = Text2Speech.from_pretrained('espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan')
    wav, _, _ = text2speech(text)
    return wav

# test_function_code --------------------

def test_convert_text_to_speech():
    """
    Test the convert_text_to_speech function.

    Raises:
        Exception: If the function does not return a torch.Tensor.
    """
    test_text = 'This is a test message.'
    result = convert_text_to_speech(test_text)
    if not isinstance(result, torch.Tensor):
        raise Exception('The function did not return a torch.Tensor.')

# call_test_function_code --------------------

test_convert_text_to_speech()