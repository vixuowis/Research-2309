from datasets import load_dataset

# Function to test the classify_spoken_digit function
# This function loads a sample from the 'mazkooleg/0-9up_google_speech_commands_augmented_raw' dataset and uses it to test the classify_spoken_digit function.
# It uses the assert statement to check if the function's output is close to the expected output.
def test_classify_spoken_digit():
    # Load the dataset
    dataset = load_dataset('mazkooleg/0-9up_google_speech_commands_augmented_raw')
    # Select a sample from the dataset
    sample = dataset['train'][0]
    # Get the expected output
    expected_output = sample['label']
    # Get the function's output
    function_output = classify_spoken_digit(sample['audio'])
    # Check if the function's output is close to the expected output
    assert torch.isclose(function_output, expected_output, atol=1e-1), 'Test failed!'
    print('Test passed!')

# Run the test function
test_classify_spoken_digit()