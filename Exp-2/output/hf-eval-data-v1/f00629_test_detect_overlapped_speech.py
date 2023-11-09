# Test function for detect_overlapped_speech
# @param None
# @return None
def test_detect_overlapped_speech():
    # Test audio file
    test_audio_file = 'test_audio.wav'
    # Call the detect_overlapped_speech function with the test audio file
    overlap_results = detect_overlapped_speech(test_audio_file)
    # Assert that the overlap_results is not None
    assert overlap_results is not None
    # Assert that the overlap_results is not empty
    assert len(overlap_results) > 0
    # Print 'All tests passed' if the above assertions pass
    print('All tests passed')

# Call the test function
test_detect_overlapped_speech()