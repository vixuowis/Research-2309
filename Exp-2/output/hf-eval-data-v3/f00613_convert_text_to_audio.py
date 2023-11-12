# function_import --------------------

import os
from huggingface_hub import unit
from fairseq import TTS

# function_code --------------------

def convert_text_to_audio(model_name: str, text: str, output_file: str):
    """
    Convert the given text to an audio file using a pre-trained Text-to-Speech model.

    Args:
        model_name (str): The name of the pre-trained TTS model.
        text (str): The text content to be converted to audio.
        output_file (str): The path to the output audio file.

    Returns:
        None
    """
    model = unit.TTS.from_pretrained(model_name)
    waveform = model.generate_audio(text)
    waveform.save(output_file)

# test_function_code --------------------

def test_convert_text_to_audio():
    """
    Test the convert_text_to_audio function.
    """
    model_name = 'facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10'
    text = 'This is a test.'
    output_file = 'test_output.wav'
    convert_text_to_audio(model_name, text, output_file)
    assert os.path.exists(output_file), 'The output audio file does not exist.'
    os.remove(output_file)
    assert not os.path.exists(output_file), 'The output audio file was not deleted.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_convert_text_to_audio()