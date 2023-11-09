import os
import numpy as np

# Function to test the classify_sentiment function
# This function uses a sample audio file to test the classify_sentiment function
# The function asserts that the output of the classify_sentiment function is a tensor

def test_classify_sentiment():
    # Path to a sample audio file
    audio_file = 'sample.wav'
    # Ensure the file exists
    assert os.path.exists(audio_file), 'Test audio file not found'
    # Call the classify_sentiment function
    result = classify_sentiment(audio_file)
    # Assert that the result is a tensor
    assert isinstance(result, torch.Tensor), 'Output is not a tensor'
    # Assert that the result is not strictly equal to any number
    assert not np.allclose(result, np.array([0, 1, 2])), 'Output is strictly equal to a number'

test_classify_sentiment()