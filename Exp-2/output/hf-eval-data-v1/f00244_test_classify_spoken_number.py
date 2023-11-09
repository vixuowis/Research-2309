# Test function for classify_spoken_number
# This function uses the torch library to load a test dataset
# It selects a sample from the dataset and uses the classify_spoken_number function to predict the spoken number
# The predicted number is then compared with the actual number using an assert statement

def test_classify_spoken_number():
    # Load the test dataset
    test_dataset = torch.load('mazkooleg/0-9up_google_speech_commands_augmented_raw')
    # Select a sample from the dataset
    sample = test_dataset[0]
    # Use the classify_spoken_number function to predict the spoken number
    predicted_number = classify_spoken_number(sample['file_path'])
    # Compare the predicted number with the actual number
    assert predicted_number == sample['label'], 'The predicted number does not match the actual number'