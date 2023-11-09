import random

# Test function for speaker_diarization
# @param test_files: List of paths to test audio files
# @return: None

def test_speaker_diarization(test_files):
    # Ensure there are test files
    if not test_files:
        raise ValueError('No test files provided')
    
    # Select a random test file
    test_file = random.choice(test_files)
    
    # Perform diarization
    diarization = speaker_diarization(test_file)
    
    # Ensure the diarization object is not None
    assert diarization is not None, 'Diarization failed'
    
    # Print success message
    print(f'Successfully diarized file: {test_file}')