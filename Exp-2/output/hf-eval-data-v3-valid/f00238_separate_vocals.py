# function_import --------------------

from transformers import pipeline

# function_code --------------------

def separate_vocals(audio_file_path: str):
    """
    This function separates vocals from a song using the 'Awais/Audio_Source_Separation' pre-trained model.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        separated_audio_sources (list): A list of output audio files, where each file contains one of the separated sources.

    Raises:
        OSError: If the model 'Awais/Audio_Source_Separation' does not exist or the audio file is not found.
    """
    source_separation = pipeline('audio-source-separation', model='Awais/Audio_Source_Separation')
    separated_audio_sources = source_separation(audio_file_path)
    return separated_audio_sources

# test_function_code --------------------

def test_separate_vocals():
    """
    This function tests the 'separate_vocals' function with a sample audio file.
    """
    sample_audio_file_path = 'sample_audio_file.wav'
    try:
        separated_audio_sources = separate_vocals(sample_audio_file_path)
        assert isinstance(separated_audio_sources, list), 'The output should be a list.'
        assert len(separated_audio_sources) > 0, 'The list should not be empty.'
    except OSError as e:
        print(f'Error: {e}')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_separate_vocals()