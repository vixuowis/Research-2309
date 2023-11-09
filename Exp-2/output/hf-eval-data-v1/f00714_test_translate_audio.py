def test_translate_audio():
    """
    This function tests the translate_audio function by translating a sample Spanish audio file to English.
    
    Returns:
    None
    """
    # Define the path to the input audio file in Spanish
    input_file = 'spanish_voice_message.wav'
    
    # Define the path where the translated audio file in English will be saved
    output_file = 'english_translation.wav'
    
    # Call the translate_audio function
    translate_audio(input_file, output_file)
    
    # Check if the output file exists
    assert os.path.exists(output_file), 'The output file does not exist.'
    
    # Check if the output file is not empty
    assert os.path.getsize(output_file) > 0, 'The output file is empty.'
    
    print('All tests passed.')

test_translate_audio()