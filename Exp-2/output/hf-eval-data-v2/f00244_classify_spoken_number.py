# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_spoken_number(audio_file_path):
    """
    Classify the spoken number in an audio file using a pre-trained model.

    Args:
        audio_file_path (str): The path to the audio file to be classified.

    Returns:
        dict: The predicted class and score.

    Raises:
        ValueError: If the audio file path is not valid.
    """
    # Import the pipeline function from the transformers library provided by Hugging Face.
    # Use the pipeline function to create an audio classification model.
    # Specify the model 'mazkooleg/0-9up-wavlm-base-plus-ft' to be loaded. This model is fine-tuned to recognize spoken numbers (0-9) in English, specifically focused on young children's voices.
    # Created classifier can be used to recognize spoken numbers from audio samples to intelligently interact with the children in the game.
    spoken_number_classifier = pipeline('audio-classification', model='mazkooleg/0-9up-wavlm-base-plus-ft')
    prediction = spoken_number_classifier(audio_file_path)
    return prediction

# test_function_code --------------------

def test_classify_spoken_number():
    """
    Test the classify_spoken_number function.
    """
    # Test with a sample audio file
    audio_file_path = 'sample_audio.wav'
    prediction = classify_spoken_number(audio_file_path)
    assert isinstance(prediction, dict), 'The prediction should be a dictionary.'
    assert 'class' in prediction, 'The prediction dictionary should have a class key.'
    assert 'score' in prediction, 'The prediction dictionary should have a score key.'

# call_test_function_code --------------------

test_classify_spoken_number()