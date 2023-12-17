# requirements_file --------------------

!pip install -U datasets transformers torch librosa

# function_import --------------------

from datasets import load_dataset
from transformers import pipeline

# function_code --------------------

def detect_emotions_from_audio(audio_file_path, top_k=1):
    """
    Detect emotions in a given audio file using a pre-trained model.

    Args:
        audio_file_path (str): The path to the audio file.
        top_k (int): Number of top probable emotions to return.

    Returns:
        list: A list of dictionaries containing detected emotions and their probabilities.
    """
    # Load the pre-trained emotion recognition model
    classifier = pipeline('audio-classification', model='superb/wav2vec2-base-superb-er')

    # Classify emotions in the audio file
    labels = classifier(audio_file_path, top_k=top_k)

    return labels

# test_function_code --------------------

def test_detect_emotions_from_audio():
    print("Testing started.")
    # Load a demo dataset with audio files
    dataset = load_dataset('anton-l/superb_demo', 'er', split='session1')

    # Test case 1: Test with the first sample from the dataset
    print("Testing case [1/1] started.")
    sample_data = dataset[0]['file']
    result = detect_emotions_from_audio(sample_data, top_k=5)
    assert isinstance(result, list) and len(result) > 0, f"Test case [1/1] failed: result should be a list with elements."
    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_detect_emotions_from_audio()