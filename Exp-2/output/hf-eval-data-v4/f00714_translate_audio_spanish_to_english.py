# requirements_file --------------------

!pip install -U fairseq

# function_import --------------------

from fairseq import pipeline

# function_code --------------------

def translate_audio_spanish_to_english(input_file, output_file):
    """
    Translates an audio message from Spanish to English using Fairseq's audio-to-audio translation model.

    :param input_file: str, A path to the input .wav file containing the Spanish voice message.
    :param output_file: str, The path where the translated English audio will be saved.
    :return: None, The translated audio is saved to the specified output file.
    """
    # Create an audio-to-audio translation pipeline
    audio_translation = pipeline('audio-to-audio-translation', model='facebook/textless_sm_sl_es')

    # Translate the Spanish audio message to English
    translated_audio = audio_translation(input_file)

    # Save the translated audio
    translated_audio.save(output_file)

    # Return nothing as the output file is saved on disk
    return None

# test_function_code --------------------

def test_translate_audio_spanish_to_english():
    print("Testing audio translation from Spanish to English.")
    # Note: In actual scenario, the audio files must be available for the tests to work.
    input_file = 'test_spanish_voice_message.wav'
    output_file = 'test_english_translation.wav'

    # Ensure the translation function does not raise exceptions
    print('Testing case [1/1] started.')
    try:
        translate_audio_spanish_to_english(input_file, output_file)
        print('Test case [1/1] passed!')
    except Exception as e:
        assert False, f'Test case failed: The translation raised an exception: {e}'
    print('Testing finished.')

# Assuming the test audio files are properly set up, you can run the test function
# test_translate_audio_spanish_to_english()