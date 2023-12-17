# requirements_file --------------------

!pip install -U datasets transformers torchaudio

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_voice_command(voice_command_file_path):
    """
    Classify a voice command audio file to determine its phrase.

    Args:
        voice_command_file_path (str): The file path to the audio file containing the voice command.

    Returns:
        dict: A dictionary with phrases as keys and their corresponding probabilities as values.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    import os
    if not os.path.exists(voice_command_file_path):
        raise FileNotFoundError(f'Audio file {voice_command_file_path} not found.')

    cmd_classifier = pipeline('audio-classification', model='superb/hubert-base-superb-ks')
    result = cmd_classifier(voice_command_file_path, top_k=2)
    probable_phrases = {'disarm security': 0.0, 'activate alarm': 0.0}

    for label in result['labels']:
        if label in probable_phrases:
            probable_phrases[label] = result['scores'][result['labels'].index(label)]
    return probable_phrases

# test_function_code --------------------

def test_classify_voice_command():
    print("Testing started.")
    # Assuming we have a dataset or mock data to load
    voice_command_file_path = "mock_data/test_audio.wav"  # Mock file path

    # Test case 1: Voice command file does not exist
    print("Testing case [1/3] started.")
    try:
        classify_voice_command("non_existent_file.wav")
        assert False, "Test case [1/3] failed: FileNotFoundError was not raised for non-existent file."
    except FileNotFoundError:
        pass  # Expected outcome

    # More test cases to be added after ensuring the environment is correctly set up
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_voice_command()