# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_spoken_command(audio_file_path):
    """
    Classify a spoken command into specific keywords using Hugging Face Transformers.

    Args:
        audio_file_path (str): The path to the audio file containing the spoken command.

    Returns:
        dict: The classified keyword and its score.

    Raises:
        ValueError: If the audio file path is not valid or the model could not be loaded.
    """
    try:
        audio_classifier = pipeline('audio-classification', model='superb/wav2vec2-base-superb-ks')
        keyword = audio_classifier(audio_file_path, top_k=1)
        return keyword
    except Exception as e:
        raise ValueError(f'Could not classify the spoken command: {e}')

# test_function_code --------------------

def test_classify_spoken_command():
    """
    Test the classify_spoken_command function with a sample audio file.
    """
    test_audio_file_path = 'path_to_test_audio_file.wav'
    # replace 'path_to_test_audio_file.wav' with the path to a test audio file
    try:
        result = classify_spoken_command(test_audio_file_path)
        assert isinstance(result, dict), 'The result should be a dictionary.'
        assert 'score' in result, 'The result should contain a score.'
        assert 'label' in result, 'The result should contain a label.'
        print('All Tests Passed')
    except Exception as e:
        print(f'Test failed: {e}')

# call_test_function_code --------------------

test_classify_spoken_command()