# function_import --------------------

from fairseq import pipeline

# function_code --------------------

def translate_audio_message(input_file: str, output_file: str) -> None:
    """
    Translates an audio message from Spanish to English using the Fairseq library.

    Args:
        input_file (str): The path to the input audio file in Spanish.
        output_file (str): The path where the translated audio file in English will be saved.

    Returns:
        None
    """
    audio_translation = pipeline('audio-to-audio-translation', model='facebook/textless_sm_sl_es')
    translated_audio = audio_translation(input_file)
    translated_audio.save(output_file)

# test_function_code --------------------

def test_translate_audio_message():
    """
    Tests the function translate_audio_message.
    """
    # Define the input and output files
    input_file = 'spanish_voice_message.wav'
    output_file = 'english_translation.wav'

    # Call the function with the test files
    translate_audio_message(input_file, output_file)

    # Check that the output file was created
    assert os.path.exists(output_file), 'The output file was not created.'

# call_test_function_code --------------------

test_translate_audio_message()