# requirements_file --------------------

!pip install -U datasets, transformers, torchaudio

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def classify_voice_command(voice_command_file_path):
    """
    Classify a voice command audio file into specific command phrases.

    Parameters:
        voice_command_file_path (str): The file path to the voice command audio file.

    Returns:
        dict: A dictionary with command phrases as keys and their corresponding probabilities as values.
    """
    cmd_classifier = pipeline('audio-classification', model='superb/hubert-base-superb-ks', top_k=2)
    result = cmd_classifier(voice_command_file_path)
    probable_actions = {'disarm security': 0.0, 'activate alarm': 0.0}
    for idx, label in enumerate(result['labels']):
        if label in probable_actions:
            probable_actions[label] = result['scores'][idx]

    return probable_actions

# test_function_code --------------------

def test_classify_voice_command():
    print("Testing classify_voice_command function.")
    # Assuming we have a dataset or sample audio files for testing
    sample_data = 'path/to/sample_voice_command.wav'  # Replace with actual file path

    # Execute the function
    results = classify_voice_command(sample_data)

    # Check if the results contain the expected keys
    assert 'disarm security' in results, f"Missing 'disarm security' in results: {results}"
    assert 'activate alarm' in results, f"Missing 'activate alarm' in results: {results}"

    # Check if the results values are floats (probabilities)
    for key, value in results.items():
        assert isinstance(value, float), f"The value for '{key}' is not a float: {value}"

    print("All tests passed successfully.")

# Run the test function
test_classify_voice_command()