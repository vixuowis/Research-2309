# function_import --------------------

from huggingface_hub import unit
from fairseq import TTS

# function_code --------------------

def convert_text_to_audio(text: str, model_name: str = 'facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10', output_file: str = 'audiobook_output.wav'):
    """
    Convert the given text into an audio file using a Text-to-Speech model.

    Args:
        text (str): The text content to be converted into audio.
        model_name (str, optional): The name of the pre-trained TTS model. Defaults to 'facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10'.
        output_file (str, optional): The name of the output audio file. Defaults to 'audiobook_output.wav'.

    Returns:
        None. The function saves the generated audio to an output file.
    """
    model = unit.TTS.from_pretrained(model_name)
    waveform = model.generate_audio(text)
    waveform.save(output_file)

# test_function_code --------------------

def test_convert_text_to_audio():
    """
    Test the convert_text_to_audio function.
    """
    test_text = 'This is a test sentence.'
    test_output_file = 'test_output.wav'
    convert_text_to_audio(test_text, output_file=test_output_file)
    assert os.path.exists(test_output_file), 'The output audio file does not exist.'
    os.remove(test_output_file)

# call_test_function_code --------------------

test_convert_text_to_audio()