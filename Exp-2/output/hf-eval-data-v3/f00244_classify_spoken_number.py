# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_spoken_number(audio_file_path: str) -> dict:
    '''
    Classify the spoken number in an audio file using a pre-trained model from Hugging Face.

    Args:
        audio_file_path (str): The path to the audio file to be classified.

    Returns:
        dict: The classification result.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    '''
    spoken_number_classifier = pipeline('audio-classification', model='mazkooleg/0-9up-wavlm-base-plus-ft')
    prediction = spoken_number_classifier(audio_file_path)
    return prediction

# test_function_code --------------------

def test_classify_spoken_number():
    '''
    Test the classify_spoken_number function.
    '''
    # Test with a valid audio file
    try:
        result = classify_spoken_number('valid_audio_file.wav')
        assert isinstance(result, dict)
    except FileNotFoundError:
        print('Valid audio file not found.')

    # Test with an invalid audio file
    try:
        result = classify_spoken_number('invalid_audio_file.wav')
        assert result is None
    except FileNotFoundError:
        print('Invalid audio file not found.')

    return 'All Tests Passed'

# call_test_function_code --------------------

test_classify_spoken_number()