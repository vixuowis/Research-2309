import torch

# Test function for 'extract_video_from_text'
# This function uses a sample text file to test the 'extract_video_from_text' function
# The function asserts that the output of the 'extract_video_from_text' function is a tensor

def test_extract_video_from_text():
    # Define the test text file
    test_text_file = 'test_text_file.txt'

    # Call the 'extract_video_from_text' function
    output = extract_video_from_text(test_text_file)

    # Assert that the output is a tensor
    assert isinstance(output, torch.Tensor), 'Output is not a tensor'