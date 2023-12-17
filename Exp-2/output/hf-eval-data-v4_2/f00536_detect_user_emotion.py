# requirements_file --------------------

!pip install -U datasets transformers torch librosa

# function_import --------------------

from transformers import pipeline
from datasets import load_dataset

# function_code --------------------

def detect_user_emotion(audio_file_path, top_k=5):
    """
    Detects emotions from an audio file using a pre-trained model.

    Args:
        audio_file_path (str): The file path to the audio file.
        top_k (int, optional): The number of top probabilities to return. Defaults to 5.

    Returns:
        list: A list of tuples containing emotion labels and their corresponding probabilities.

    Raises:
        FileNotFoundError: If the audio_file_path does not exist.
        ValueError: If the audio_file_path is not an audio file.
    """
    # Instantiate the emotion recognition classifier
    classifier = pipeline('audio-classification', model='superb/wav2vec2-base-superb-er')
    
    # Perform emotion detection
    try:
        labels = classifier(audio_file_path, top_k=top_k)
    except Exception as e:
        raise e
    
    return labels

# test_function_code --------------------

def test_detect_user_emotion():
    print("Testing started.")
    
    # Load a sample dataset for testing
    dataset = load_dataset('anton-l/superb_demo', 'er', split='session1')
    audio_file_path = dataset[0]['file']

    # Test case 1: Testing with a valid audio file
    print("Testing case [1/1] started.")
    labels = detect_user_emotion(audio_file_path, top_k=3)
    assert labels and len(labels) <= 3, f"Test case [1/1] failed: Expected top 3 emotions, got {len(labels)} labels."
    print("Testing finished.")

# Call the test function


# call_test_function_line --------------------

test_detect_user_emotion()