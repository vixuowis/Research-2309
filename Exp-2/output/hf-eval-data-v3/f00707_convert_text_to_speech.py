# function_import --------------------

import torch
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech

# function_code --------------------

def convert_text_to_speech(text):
    '''
    Convert a given text into synthesized speech using a pretrained Text-to-Speech model.

    Args:
        text (str): The text message to be converted into speech.

    Returns:
        numpy.ndarray: The synthesized speech in the form of a waveform.

    Raises:
        ModuleNotFoundError: If the necessary modules are not found.
    '''
    d = ModelDownloader()
    text2speech = Text2Speech.from_pretrained('espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan')
    wav, _, _ = text2speech(text)
    return wav

# test_function_code --------------------

def test_convert_text_to_speech():
    '''
    Test the convert_text_to_speech function.
    '''
    text = 'Hello, world!'
    wav = convert_text_to_speech(text)
    assert isinstance(wav, np.ndarray), 'The output should be a numpy array.'
    assert wav.shape[0] > 0, 'The output waveform should not be empty.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_convert_text_to_speech()