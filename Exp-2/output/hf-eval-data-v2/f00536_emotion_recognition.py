# function_import --------------------

from datasets import load_dataset
from transformers import pipeline

# function_code --------------------

def emotion_recognition(audio_file, top_k=5):
    """
    This function classifies the emotion in a given audio file using a pre-trained model.

    Args:
        audio_file (str): The path to the audio file to be classified.
        top_k (int, optional): The number of top predictions to return. Defaults to 5.

    Returns:
        list: A list of tuples containing the predicted emotion and the corresponding score.
    """
    # Load the dataset
    dataset = load_dataset('anton-l/superb_demo', 'er', split='session1')
    # Create the classifier
    classifier = pipeline('audio-classification', model='superb/wav2vec2-base-superb-er')
    # Classify the emotion in the audio file
    labels = classifier(audio_file, top_k=top_k)
    return labels

# test_function_code --------------------

def test_emotion_recognition():
    """
    This function tests the emotion_recognition function.
    """
    # Define the path to the test audio file
    test_audio_file = 'path_to_test_audio_file'
    # Call the emotion_recognition function
    result = emotion_recognition(test_audio_file)
    # Assert that the result is a list
    assert isinstance(result, list)
    # Assert that the length of the result is equal to the top_k value
    assert len(result) == 5
    # Assert that each element in the result is a tuple
    for label in result:
        assert isinstance(label, tuple)
        # Assert that the first element of the tuple is a string (the predicted emotion)
        assert isinstance(label[0], str)
        # Assert that the second element of the tuple is a float (the score)
        assert isinstance(label[1], float)

# call_test_function_code --------------------

test_emotion_recognition()