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
        Exception: If the audio file could not be loaded or the language could not be identified.
    """
    try:
        # Load the language identification model
        language_id = EncoderClassifier.from_hparams(source='speechbrain/lang-id-voxlingua107-ecapa', savedir='/tmp')
        # Load the audio file
        signal = language_id.load_audio(audio_url)
        # Identify the language
        prediction = language_id.classify_batch(signal)
        return prediction
    except Exception as e:
        print(f'An error occurred: {e}')

# test_function_code --------------------

def test_identify_language():
    """
    Test the identify_language function.
    """
    # Test audio file URL
    test_url = 'https://omniglot.com/soundfiles/udhr/udhr_th.mp3'
    # Expected result (Thai language)
    expected_result = 'th'
    # Get the prediction
    prediction = identify_language(test_url)
    # Check if the prediction is close to the expected result
    assert prediction == expected_result, f'Expected {expected_result}, but got {prediction}'

# call_test_function_code --------------------

test_identify_language()