# requirements_file --------------------

!pip install -U transformers asteroid

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def separate_vocals(audio_file_path):
    """
    Separate the vocal track from a given audio file.

    Args:
        audio_file_path (str): The file path to the audio file to be processed.

    Returns:
        list: An array of output audio files, where each contains a separate source.

    Raises:
        FileNotFoundError: If the input file path does not exist.
        ValueError: If the input is not a valid audio file.
    """
    source_separation = pipeline('audio-source-separation', model='Awais/Audio_Source_Separation')
    return source_separation(audio_file_path)

# test_function_code --------------------

import os

# Assume 'examples/audio_sample.mp3' is a valid audio file for testing purposes
def test_separate_vocals():
    print("Testing started.")
    audio_file_path = 'examples/audio_sample.mp3'

    if not os.path.isfile(audio_file_path):
        raise FileNotFoundError(f"Test file {audio_file_path} not found.")

    # Testing case: Separate vocals from the audio file
    print("Testing case [1/1] started.")
    separated_sources = separate_vocals(audio_file_path)
    assert isinstance(separated_sources, list), "Test case failed: The result is not a list of audio files."
    print("Testing finished.")

# Run the test function
if __name__ == '__main__':
    test_separate_vocals()

# call_test_function_line --------------------

test_separate_vocals()