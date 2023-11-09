def test_speaker_identification():
    """
    This function is used to test the speaker_identification function.
    It uses a test dataset from the VoxCeleb1 dataset.
    The test is not strict number comparison, it checks if the function returns a list of length 5.
    """
    # Load the test dataset
    from datasets import load_dataset
    dataset = load_dataset('anton-l/superb_demo', 'si', split='test')
    
    # Select a sample from the dataset
    audio_file_path = dataset[0]['file']
    
    # Call the function with the sample
    result = speaker_identification(audio_file_path)
    
    # Test if the function returns a list of length 5
    assert len(result) == 5, 'The function should return a list of length 5.'
    
    print('Test passed.')

# Run the test function
test_speaker_identification()