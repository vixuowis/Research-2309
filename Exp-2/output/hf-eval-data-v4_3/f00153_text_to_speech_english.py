# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def text_to_speech_english(sentence, output_path):
    """
    Convert an English sentence to speech using the ESPnet mio/Artoria Text-to-Speech model.
    
    Args:
        sentence (str): The English sentence to be translated into speech.
        output_path (str): The file path where the speech audio will be saved.

    Returns:
        str: The file path where the audio is saved, as confirmation.

    Raises:
        ValueError: If the inputs are not valid strings.
        IOError: If there is an error saving the audio file.
    """
    if not isinstance(sentence, str) or not isinstance(output_path, str):
        raise ValueError("Both input sentence and output path must be strings.")

    # Initialize the pipeline with the specified model.
    tts_pipeline = pipeline('text-to-speech', model='mio/Artoria')

    # Generate the speech audio for the input sentence.
    speech_output = tts_pipeline(sentence)

    # Save the speech audio to the specified file path.
    with open(output_path, 'wb') as audio_file:
        audio_file.write(speech_output['wav'])

    return output_path

# test_function_code --------------------

import os

def test_text_to_speech_english():
    print("Testing started.")

    # Prepare sample data for testing.
    sample_sentence = "Hello, world!"
    output_audio_path = "output_test_audio.wav"

    # Test case 1: Valid input sentence and output path.
    print("Testing case [1/3] started.")
    result_path = text_to_speech_english(sample_sentence, output_audio_path)
    assert os.path.exists(result_path), f"Test case [1/3] failed: Audio file was not saved at {result_path}."

    # Cleanup the generated audio file.
    if os.path.exists(output_audio_path):
        os.remove(output_audio_path)
    
    # Test case 2: Invalid input sentence (not a string).
    print("Testing case [2/3] started.")
    try:
        text_to_speech_english(123, output_audio_path)
        assert False, "Test case [2/3] failed: ValueError should have been raised for non-string sentence."
    except ValueError:
        pass  # Expected exception.
    
    # Test case 3: Invalid output path (not a string).
    print("Testing case [3/3] started.")
    try:
        text_to_speech_english(sample_sentence, 456)
        assert False, "Test case [3/3] failed: ValueError should have been raised for non-string output path."
    except ValueError:
        pass  # Expected exception.

    print("Testing finished.")

# Run the test function.
test_text_to_speech_english()

# call_test_function_line --------------------

test_text_to_speech_english()