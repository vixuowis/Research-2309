# requirements_file --------------------

!pip install -U transformers datasets torchaudio

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_keyword_in_audio(audio_file_path, top_k=5):
    """
    Classify the keyword spoken in a recorded audio file.

    Parameters:
        audio_file_path (str): The path to the audio file to be classified.
        top_k (int): The number of top predictions to return. Default is 5.

    Returns:
        list: The predicted keywords and their associated probabilities.
    """
    # Initialize the audio classification pipeline
    classifier = pipeline('audio-classification', model='superb/hubert-base-superb-ks')
    
    # Classify and get top-k keyword predictions for the audio file
    keyword_predictions = classifier(audio_file_path, top_k=top_k)
    return keyword_predictions

# test_function_code --------------------

def test_classify_keyword_in_audio():
    print("Testing started.")
    audio_file_path = 'audio_sample_16khz.wav'  # Sample audio file

    # Test case 1: Check if the function returns a list
    print("Testing case [1/1] started.")
    predictions = classify_keyword_in_audio(audio_file_path)
    assert isinstance(predictions, list), f"Test case [1/1] failed: The function should return a list, got {type(predictions)} instead."
    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_classify_keyword_in_audio()