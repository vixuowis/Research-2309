# function_import --------------------

from speechbrain.pretrained import EncoderClassifier, load_audio

# function_code --------------------

def detect_spoken_language(audio_file_path):
    """
    Detects the language being spoken in an audio file.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        str: The predicted language.
    """
    # Initialize the language identification model
    language_id = EncoderClassifier.from_hparams(source='speechbrain/lang-id-voxlingua107-ecapa', savedir='/tmp')
    # Load the audio samples from the file
    signal = load_audio(audio_file_path)
    # Process the audio samples and predict the spoken language
    prediction = language_id.classify_batch(signal)
    return prediction

# test_function_code --------------------

def test_detect_spoken_language():
    """
    Tests the detect_spoken_language function.
    """
    # Test audio file path
    test_audio_file_path = 'https://omniglot.com/soundfiles/udhr/udhr_th.mp3'
    # Expected output
    expected_output = 'th'
    # Call the detect_spoken_language function
    output = detect_spoken_language(test_audio_file_path)
    # Assert that the output is as expected
    assert output == expected_output, f'Expected {expected_output}, but got {output}'

# call_test_function_code --------------------

test_detect_spoken_language()