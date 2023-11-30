# function_import --------------------

from transformers import pipeline, Wav2Vec2ForCTC

# function_code --------------------

def classify_audio(audio_file_path):
    """
    Classify the category of the audio file using a pre-trained model.

    Args:
        audio_file_path (str): The path to the audio file to be classified.

    Returns:
        str: The category of the audio file.

    Raises:
        OSError: If there is a problem with the file path or the file itself.
    """
    print("classify_audio", audio_file_path)
    model = pipeline('audio-classification', device=0, model='superb/wav2vec2-base-superb-speech-emotion-recognition')
    result = model(audio_file_path)
    return result[0]['label']


# test_function_code --------------------

def test_classify_audio():
    """
    Test the classify_audio function with a sample audio file.

    Returns:
        str: 'All Tests Passed' if all assertions pass, otherwise the error message.
    """
    sample_audio_file_path = 'sample_audio.wav'
    try:
        category = classify_audio(sample_audio_file_path)
        assert isinstance(category, str), 'The output should be a string.'
    except Exception as e:
        return str(e)
    return 'All Tests Passed'


# call_test_function_code --------------------

test_classify_audio()