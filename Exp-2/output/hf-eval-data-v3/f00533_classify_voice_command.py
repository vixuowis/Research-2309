# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_voice_command(voice_command_file_path: str, top_k: int = 2) -> dict:
    '''
    Classify the voice command into specific phrases.

    Args:
        voice_command_file_path (str): The file path of the voice command.
        top_k (int, optional): The number of top probable actions to return. Defaults to 2.

    Returns:
        dict: A dictionary of probable actions and their scores.
    '''
    cmd_classifier = pipeline('audio-classification', model='superb/hubert-base-superb-ks')
    result = cmd_classifier(voice_command_file_path, top_k=top_k)
    probable_actions = {'disarm security': 0.0, 'activate alarm': 0.0}
    for label in result['labels']:
        if label in probable_actions:
            probable_actions[label] = result['scores'][result['labels'].index(label)]
    return probable_actions

# test_function_code --------------------

def test_classify_voice_command():
    '''
    Test the function classify_voice_command.
    '''
    # Test case 1: Test with a voice command file path
    result = classify_voice_command('test_voice_command.wav')
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'disarm security' in result, 'The result should contain the action disarm security.'
    assert 'activate alarm' in result, 'The result should contain the action activate alarm.'

    # Test case 2: Test with a voice command file path and top_k
    result = classify_voice_command('test_voice_command.wav', top_k=3)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'disarm security' in result, 'The result should contain the action disarm security.'
    assert 'activate alarm' in result, 'The result should contain the action activate alarm.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_voice_command()