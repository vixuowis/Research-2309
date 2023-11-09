from datasets import load_dataset

# Function to test the identify_spoken_language function
# Uses the google/xtreme_s dataset for testing
# Asserts that the function returns a valid language ID

def test_identify_spoken_language():
    # Load the google/xtreme_s dataset
    dataset = load_dataset('google/xtreme_s')
    # Select a sample from the dataset
    sample = dataset['test'][0]
    # Get the audio file path from the sample
    audio_file_path = sample['file']
    # Call the identify_spoken_language function
    predicted_language_id = identify_spoken_language(audio_file_path)
    # Assert that the function returns a valid language ID
    assert isinstance(predicted_language_id, int) and predicted_language_id >= 0