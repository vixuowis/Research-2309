def test_identify_speaker():
    '''
    This function tests the identify_speaker function by loading a test dataset and comparing the output with the expected result.
    The test will pass if the output and expected result are close enough, considering a small margin of error due to the probabilistic nature of the model.
    '''
    test_dataset = load_dataset('VoxCeleb1')
    speaker_identity = identify_speaker('VoxCeleb1')
    for audio_file in test_dataset:
        assert abs(speaker_identity[audio_file['file']][0]['score'] - 0.9035) < 0.1, 'Test failed!'
    print('Test passed!')

test_identify_speaker()