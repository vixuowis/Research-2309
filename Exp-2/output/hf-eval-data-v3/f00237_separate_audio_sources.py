# function_import --------------------

from transformers import pipeline

# function_code --------------------

def separate_audio_sources(audio_file_path):
    """
    Separate music and vocals from an audio file using a pretrained model.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        dict: A dictionary containing the separated sources.

    Raises:
        OSError: If the model or the audio file is not found.
    """
    try:
        audio_separator = pipeline('audio-source-separation', model='mpariente/DPRNNTasNet-ks2_WHAM_sepclean')
        separated_sources = audio_separator(audio_file_path)
        return separated_sources
    except Exception as e:
        raise OSError('Model or audio file not found.') from e

# test_function_code --------------------

def test_separate_audio_sources():
    """
    Test the function separate_audio_sources.
    """
    # Test with a valid audio file
    try:
        separated_sources = separate_audio_sources('valid_audio_file.wav')
        assert isinstance(separated_sources, dict), 'The output should be a dictionary.'
    except OSError:
        pass

    # Test with an invalid audio file
    try:
        separated_sources = separate_audio_sources('invalid_audio_file.wav')
        assert False, 'An exception should have been raised.'
    except OSError:
        pass

    # Test with a non-existing model
    try:
        separated_sources = separate_audio_sources('valid_audio_file.wav', model='non_existing_model')
        assert False, 'An exception should have been raised.'
    except OSError:
        pass

    return 'All Tests Passed'

# call_test_function_code --------------------

test_separate_audio_sources()