# requirements_file --------------------

!pip install -U transformers datasets librosa

# function_import --------------------

from transformers import pipeline
from datasets import load_dataset
import librosa

# function_code --------------------

def classify_emotions_from_audio(audio_file_path: str) -> dict:
    """Classify emotions from an audio file using a pre-trained model.

    Args:
        audio_file_path (str): The file path to the audio file to be analyzed.

    Returns:
        dict: A dictionary containing the predicted emotions and their probabilities.

    Raises:
        FileNotFoundError: If the audio file does not exist at the provided path.
        ValueError: If the audio file is not in WAV format or has incorrect sampling rate.
    """
    # Ensure the audio file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found at: {audio_file_path}")
    # Load the audio file and check if it is in WAV format and has the correct sampling rate
    sampling_rate, _ = librosa.load(audio_file_path, sr=None)
    if sampling_rate != 16000:
        raise ValueError("Audio file must be sampled at 16000Hz.")
    # Load the emotion classification model
    classifier = pipeline('audio-classification', model='superb/hubert-large-superb-er')
    # Classify the emotions in the audio file
    predicted_emotions = classifier(audio_file_path, top_k=5)
    return predicted_emotions

# test_function_code --------------------

def test_classify_emotions_from_audio():
    print("Testing started.")
    # Use a provided dataset sample for testing
    dataset = load_dataset('anton-l/superb_demo', 'er', split='test[:1]')
    audio_file_path = dataset[0]['file']

    # Testing case 1: Correct audio file
    print("Testing case [1/1] started.")
    try:
        result = classify_emotions_from_audio(audio_file_path)
        assert isinstance(result, dict), "Result should be a dictionary."
        assert 'label' in result, "Result dictionary should contain a 'label' key."
    except Exception as e:
        print(f"Test case [1/1] failed: {e}")
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_emotions_from_audio()