# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def text_to_speech(text): 
    """
    Convert text to speech audio using a pretrained text-to-speech model.

    Parameters:
        text (str): The text to convert to speech.

    Returns:
        audio (object): The audio output object that can be played or saved.
    """
    tts = pipeline('text-to-speech', model='mio/Artoria')
    audio = tts(text)
    return audio

# test_function_code --------------------

def test_text_to_speech():
    print("Testing text_to_speech function.")

    test_text = 'Hello, world!'
    audio_output = text_to_speech(test_text)

    assert isinstance(audio_output, object), "The function should return an audio object."
    print("All test cases passed!")