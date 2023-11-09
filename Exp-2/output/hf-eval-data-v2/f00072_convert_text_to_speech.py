# function_import --------------------

from transformers import pipeline
import soundfile as sf

# function_code --------------------

def convert_text_to_speech(text: str, output_file: str = 'output.wav'):
    """
    Convert a given Japanese sentence into a speech audio file.

    Args:
        text (str): The Japanese text to be converted to speech.
        output_file (str, optional): The name of the output audio file. Defaults to 'output.wav'.

    Returns:
        None. The function saves the audio file to the current directory.
    """
    tts = pipeline('text-to-speech', model='espnet/kan_bayashi_jvs_tts_finetune_jvs001_jsut_vits_raw_phn_jaconv_pyopenjta-truncated-178804')
    audio_waveform = tts(text)[0]['generated_sequence']
    sf.write(output_file, audio_waveform, samplerate=24000)

# test_function_code --------------------

def test_convert_text_to_speech():
    """
    Test the convert_text_to_speech function.

    The function does not return any value. Therefore, the test will be to check if the output file is created.
    """
    import os
    convert_text_to_speech('こんにちは、世界', 'test_output.wav')
    assert os.path.exists('test_output.wav'), 'The audio file was not created.'

# call_test_function_code --------------------

test_convert_text_to_speech()