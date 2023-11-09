# Test function for speaker_diarization
# This function will test the speaker_diarization function using a sample audio file.
def test_speaker_diarization():
    # Define the path to the sample audio file
    audio_file = 'sample_audio.wav'
    
    # Call the speaker_diarization function
    output_file = speaker_diarization(audio_file)
    
    # Check if the output file exists
    assert os.path.isfile(output_file), f'Output file {output_file} not found'
    
    # Check if the output file is not empty
    assert os.path.getsize(output_file) > 0, 'Output file is empty'
    
    print('All tests passed.')

# Run the test function
test_speaker_diarization()