# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_audio(audio_input):
    """
    Classify the demographics of a caller based on their voice.

    Args:
        audio_input (str): The path to the audio file to be classified.

    Returns:
        dict: The classification result.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    classifier = pipeline('audio-classification', model='superb/wav2vec2-base-superb-sid')
    result = classifier(audio_input)
    return result

# test_function_code --------------------

def test_classify_audio():
    """
    Test the classify_audio function.
    """
    test_audio = 'test.wav'
    try:
        result = classify_audio(test_audio)
        assert isinstance(result, dict)
        print('Test passed.')
    except FileNotFoundError:
        print('Test audio file not found.')

# call_test_function_code --------------------

test_classify_audio()