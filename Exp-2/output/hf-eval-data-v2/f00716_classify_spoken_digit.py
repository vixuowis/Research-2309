# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_spoken_digit(audio_sample_path):
    """
    Classify the spoken digit in the provided audio sample using a pre-trained model.

    Args:
        audio_sample_path (str): The path to the audio sample to be classified.

    Returns:
        dict: The predicted digit and the score of the prediction.
    """
    spoken_digit_classifier = pipeline('audio-classification', model='MIT/ast-finetuned-speech-commands-v2')
    digit_prediction = spoken_digit_classifier(audio_sample_path)
    return digit_prediction

# test_function_code --------------------

def test_classify_spoken_digit():
    """
    Test the classify_spoken_digit function with a sample audio file.
    """
    sample_audio_path = 'path_to_sample_audio_file'
    prediction = classify_spoken_digit(sample_audio_path)
    assert isinstance(prediction, dict), 'The prediction should be a dictionary.'
    assert 'label' in prediction, 'The prediction dictionary should have a label key.'
    assert 'score' in prediction, 'The prediction dictionary should have a score key.'

# call_test_function_code --------------------

test_classify_spoken_digit()