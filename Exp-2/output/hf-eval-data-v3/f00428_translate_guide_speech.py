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

    Raises:
        ModuleNotFoundError: If the 'fairseq' library is not installed.
    """
    s2s_translation_model = textless_sm_sl_es()
    translated_audio = s2s_translation_model(audio_input)
    return translated_audio

# test_function_code --------------------

def test_translate_guide_speech():
    """
    Tests the 'translate_guide_speech' function by providing it with an audio input and checking if it returns an audio output.
    """
    # Test case: Provide an audio input and check if it returns an audio output.
    audio_input = 'test_audio.wav'  # Replace this with a real audio file for testing.
    translated_audio = translate_guide_speech(audio_input)
    assert isinstance(translated_audio, type(audio_input)), 'The translated audio output is not of the same type as the input.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_guide_speech()