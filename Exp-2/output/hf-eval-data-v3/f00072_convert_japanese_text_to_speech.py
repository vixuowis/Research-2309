# function_import --------------------

from transformers import pipeline
import soundfile as sf
import os

# function_code --------------------

def convert_japanese_text_to_speech(text: str, output_file: str) -> None:
    """
    Convert a given Japanese sentence into a speech audio file.

    Args:
        text (str): The Japanese text to be converted to speech.
        output_file (str): The path of the output audio file.

    Returns:
        None

    Raises:
        Exception: If the text is not a string or the output_file is not a string.
    """
    if not isinstance(text, str):
        raise Exception('The text should be a string.')
    if not isinstance(output_file, str):
        raise Exception('The output_file should be a string.')

    tts = pipeline('text-to-speech', model='espnet/kan-bayashi_jvs_tts_finetune_jvs001_jsut_vits_raw_phn_jaconv_pyopenjta-truncated-178804')
    audio_waveform = tts(text)[0]['generated_sequence']
    sf.write(output_file, audio_waveform, samplerate=24000)

# test_function_code --------------------

def test_convert_japanese_text_to_speech():
    """
    Test the convert_japanese_text_to_speech function.
    """
    # Test with valid inputs
    convert_japanese_text_to_speech('こんにちは、世界', 'output.wav')
    assert os.path.exists('output.wav'), 'The output file does not exist.'

    # Test with invalid inputs
    try:
        convert_japanese_text_to_speech(123, 'output.wav')
    except Exception as e:
        assert str(e) == 'The text should be a string.', 'The exception message is incorrect.'

    try:
        convert_japanese_text_to_speech('こんにちは、世界', 123)
    except Exception as e:
        assert str(e) == 'The output_file should be a string.', 'The exception message is incorrect.'

    print('All Tests Passed')

# call_test_function_code --------------------

test_convert_japanese_text_to_speech()