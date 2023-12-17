# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

import torch
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.tts_inference import Text2Speech

# function_code --------------------

def convert_text_to_speech(text):
    """
    Convert the input text to speech using a pretrained Text-to-Speech model from ESPnet.

    Parameters:
        text (str): The text message to be converted into speech.

    Returns:
        numpy.ndarray: The waveform of the synthesized speech audio.
    """
    # Initialize the downloader for the pretrained model
    d = ModelDownloader()

    # Load the pretrained Text-to-Speech model
    text2speech = Text2Speech.from_pretrained('espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan')

    # Convert the text to speech
    wav, _, _ = text2speech(text)

    # Return the waveform
    return wav.numpy()

# test_function_code --------------------

def test_convert_text_to_speech():
    print("Testing the convert_text_to_speech function.")

    # Define a simple test case
    test_text = "Hello, world! This is a test message."
    expected_length = 100 # Assuming a specific length for the test

    # Call the function with the test case
    wav = convert_text_to_speech(test_text)

    # Check if the output is as expected
    assert wav.shape[0] > expected_length, f"Expected waveform length to be greater than {expected_length}, but got {wav.shape[0]} instead."

    print("Passed the test.")

# Run the test function
test_convert_text_to_speech()