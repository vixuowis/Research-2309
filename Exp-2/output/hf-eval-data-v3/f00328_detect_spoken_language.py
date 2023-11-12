# function_import --------------------

import torch
from speechbrain.pretrained import EncoderClassifier, load_audio

# function_code --------------------

def detect_spoken_language(audio_file_path):
    """
    Detects the language being spoken in an audio file.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        str: The predicted language.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    # Initialize the language identification model
    language_id = EncoderClassifier.from_hparams(source='speechbrain/lang-id-voxlingua107-ecapa', savedir='/tmp')
    # Load the audio file
    signal = load_audio(audio_file_path)
    # Predict the spoken language
    prediction = language_id.classify_batch(signal)
    return prediction

# test_function_code --------------------

def test_detect_spoken_language():
    """Tests the detect_spoken_language function."""
    # Test with a Thai language audio file
    assert detect_spoken_language('https://omniglot.com/soundfiles/udhr/udhr_th.mp3') == 'th'
    # Test with a English language audio file
    assert detect_spoken_language('https://omniglot.com/soundfiles/udhr/udhr_en.mp3') == 'en'
    # Test with a non-existent file
    try:
        detect_spoken_language('non_existent_file.mp3')
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Expected a FileNotFoundError.')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_spoken_language()