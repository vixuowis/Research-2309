# requirements_file --------------------

!pip install -U transformers, datasets

# function_import --------------------

from transformers import pipeline
from datasets import load_dataset

# function_code --------------------

def estimate_caller_demographics(audio_input):
    # Create a pipeline for audio classification using the Hugging Face Transformers library
    classifier = pipeline('audio-classification', model='superb/wav2vec2-base-superb-sid')
    # Classify the audio input to estimate the demographics of the caller
    result = classifier(audio_input)
    return result

# test_function_code --------------------

def test_estimate_caller_demographics():
    print("Testing started.")
    # Load a sample dataset
    dataset = load_dataset('anton-l/superb_demo', 'si', split='test')
    # Get a sample audio input from the dataset
    sample_data = dataset[0]['file']

    # Call the function with the sample audio
    print("Testing case started.")
    result = estimate_caller_demographics(sample_data)
    # Check if the result is not empty
    assert result, "Test case failed: The function did not return any result."
    print("Testing finished.")

# Run the test function
test_estimate_caller_demographics()