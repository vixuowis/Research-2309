# function_import --------------------

import fairseq
from fairseq.models import textless_sm_sl_es

# function_code --------------------

def translate_guide_speech(audio_input):
    """
    Translates the audio input from a guide's speech to Spanish using the 'textless_sm_sl_es' model from Fairseq.

    Args:
        audio_input (AudioData): The audio input from the guide's speech.

    Returns:
        AudioData: The translated audio output in Spanish.
    """
    s2s_translation_model = textless_sm_sl_es()
    translated_audio = s2s_translation_model(audio_input)
    return translated_audio

# test_function_code --------------------

def test_translate_guide_speech():
    """
    Tests the 'translate_guide_speech' function by using a sample audio input and checking if the output is not None.
    """
    sample_audio_input = 'sample_audio_input.wav'  # Replace with a real audio file for testing
    translated_audio = translate_guide_speech(sample_audio_input)
    assert translated_audio is not None, 'The translation function returned None.'

# call_test_function_code --------------------

test_translate_guide_speech()