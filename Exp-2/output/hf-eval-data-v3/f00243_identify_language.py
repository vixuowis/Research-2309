# function_import --------------------

from speechbrain.pretrained import EncoderClassifier
import torchaudio

# function_code --------------------

def identify_language(audio_url):
    """
    Identify the language spoken in an audio file.

    Args:
        audio_url (str): The URL of the audio file to be analyzed.

    Returns:
        str: The predicted language.

    Raises:
        Exception: If the audio file could not be processed.
    """
    try:
        language_id = EncoderClassifier.from_hparams(source='speechbrain/lang-id-voxlingua107-ecapa', savedir='/tmp')
        signal = language_id.load_audio(audio_url)
        prediction = language_id.classify_batch(signal)
        return prediction
    except Exception as e:
        raise Exception('Failed to process the audio file.') from e

# test_function_code --------------------

def test_identify_language():
    """
    Test the identify_language function.
    """
    # Test with a Thai language audio file
    assert identify_language('https://omniglot.com/soundfiles/udhr/udhr_th.mp3') == 'th'
    # Test with a English language audio file
    assert identify_language('https://omniglot.com/soundfiles/udhr/udhr_en.mp3') == 'en'
    # Test with a Spanish language audio file
    assert identify_language('https://omniglot.com/soundfiles/udhr/udhr_es.mp3') == 'es'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_identify_language()