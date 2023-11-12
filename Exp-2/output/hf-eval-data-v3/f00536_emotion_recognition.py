# function_import --------------------

import torch
from datasets import load_dataset
from transformers import pipeline

# function_code --------------------

def emotion_recognition(audio_file: str, top_k: int = 5):
    """
    Function to detect emotions from an audio file using a pre-trained model.

    Args:
        audio_file (str): Path to the audio file.
        top_k (int, optional): Number of top predictions to return. Defaults to 5.

    Returns:
        list: List of dictionaries containing 'label' and 'score' for top_k predictions.

    Raises:
        ImportError: If necessary libraries are not installed.
    """
    dataset = load_dataset('anton-l/superb_demo', 'er', split='session1')
    classifier = pipeline('audio-classification', model='superb/wav2vec2-base-superb-er')
    labels = classifier(audio_file, top_k=top_k)
    return labels

# test_function_code --------------------

def test_emotion_recognition():
    """
    Function to test the emotion_recognition function.
    """
    test_audio_file = 'test_audio.wav'
    result = emotion_recognition(test_audio_file)
    assert isinstance(result, list), 'Result should be a list.'
    assert len(result) > 0, 'Result list should not be empty.'
    for label in result:
        assert 'label' in label, 'Each item should have a label.'
        assert 'score' in label, 'Each item should have a score.'
    print('All Tests Passed')

# call_test_function_code --------------------

test_emotion_recognition()