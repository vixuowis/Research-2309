# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def separate_vocals_and_instrumentals(audio_file_path):
    """
    Separate the vocals and instrumentals from an audio file using a pretrained model.

    Args:
        audio_file_path (str): The path to the audio file that needs to be processed.

    Returns:
        dict: contains separated audio sources as 'vocals' and 'background'.

    Raises:
        ValueError: If 'audio_file_path' is not a valid path or is None.
    """
    if audio_file_path is None or not isinstance(audio_file_path, str):
        raise ValueError("Invalid audio file path.")

    audio_separator = pipeline('audio-source-separation', model='mpariente/DPRNNTasNet-ks2_WHAM_sepclean')
    separated_sources = audio_separator(audio_file_path)

    return {
        'vocals': separated_sources[0],
        'background': separated_sources[1]
    }

# test_function_code --------------------

import os

def test_separate_vocals_and_instrumentals():
    print("Testing started.")
    sample_audio = 'audio_sample.wav'  # Will require a sample audio file in the test environment

    # Check if the sample audio file exists
    assert os.path.isfile(sample_audio), "Test case failed: Sample audio file does not exist."

    # Testing case 1: Separate vocals and instruments
    print("Testing case [1/1] started.")
    output = separate_vocals_and_instrumentals(sample_audio)
    assert 'vocals' in output and 'background' in output, "Test case [1/1] failed: Keys 'vocals' and 'background' are missing from the output."
    print("Testing finished.")

# call_test_function_line --------------------

test_separate_vocals_and_instrumentals()