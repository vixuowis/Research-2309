# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import Text2Speech
import torch

# function_code --------------------

def create_audiobook(text):
    """
    Converts input text to speech using a pretrained ESPnet model.

    Parameters:
        text (str): The text content to be converted to speech.

    Returns:
        torch.Tensor: The synthesized speech output as an audio waveform.
    """
    # Load the pretrained Text-to-Speech model
    model = Text2Speech.from_pretrained('espnet/kan-bayashi_ljspeech_joint_finetune_conformer_fastspeech2_hifigan')

    # Convert text to speech
    speech_output = model(text)

    return speech_output


# test_function_code --------------------

def test_create_audiobook():
    print("Testing create_audiobook function.")

    # Test case 1: Empty string
    print("Testing with empty string.")
    speech_output = create_audiobook("")
    assert speech_output is not None, "Empty string should still produce some audio output."

    # Test case 2: Short text
    print("Testing with short text.")
    speech_output = create_audiobook("Hello, world!")
    assert isinstance(speech_output, torch.Tensor), "Output should be a torch tensor."

    # Test case 3: Long text
    print("Testing with long text.")
    speech_output = create_audiobook("This is a longer text to check if the function can handle more content.")
    assert speech_output.shape[0] > 0, "Long text should yield non-empty audio waveform."

    print("All tests passed!")

# Run the test function
test_create_audiobook()
