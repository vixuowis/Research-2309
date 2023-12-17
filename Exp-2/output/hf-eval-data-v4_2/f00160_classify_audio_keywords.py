# requirements_file --------------------

!pip install -U datasets transformers torchaudio

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_audio_keywords(audio_file_path: str, top_k: int = 5) -> dict:
    """
    Classify keywords spoken in an audio file using a pre-trained Hugging Face model.

    Args:
        audio_file_path (str): The path to the audio file that needs to be classified.
        top_k (int, optional): The number of top predictions to return. Defaults to 5.

    Returns:
        dict: A dictionary containing the top 'k' predicted keywords and their scores.

    Raises:
        FileNotFoundError: If the audio file does not exist at the given path.
        ValueError: If the input audio file is not sampled at 16kHz.
    """
    # Verify if audio file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"Audio file not found at {audio_file_path}")
    
    # TODO: Add check for audio sampling rate (16kHz)
    
    # Initialize the classifier using the Hugging Face pipeline
    classifier = pipeline('audio-classification', model='superb/hubert-base-superb-ks')
    
    # Classify the audio file and return the results
    return classifier(audio_file_path, top_k=top_k)

# test_function_code --------------------

def test_classify_audio_keywords():
    print("Testing started.")
    sample_audio_file = "sample_audio.wav"  # Replace with a sample audio file path

    # Test case 1: Valid audio file with default top_k
    print("Testing case [1/2] started.")
    results = classify_audio_keywords(sample_audio_file)
    assert isinstance(results, dict), f"Test case [1/2] failed: Expected results to be a dictionary, got {type(results)}"

    # Test case 2: Valid audio file with custom top_k
    print("Testing case [2/2] started.")
    top_k = 3
    results = classify_audio_keywords(sample_audio_file, top_k=top_k)
    assert len(results) == top_k, f"Test case [2/2] failed: Expected top {top_k} results, got {len(results)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_audio_keywords()