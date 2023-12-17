# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

from speechbrain.pretrained import EncoderClassifier
import torchaudio

# function_code --------------------

def identify_language_from_audio_url(audio_url: str) -> str:
    """
    Identifies the language spoken in the audio file located at the given URL.

    Args:
        audio_url (str): The URL of the audio file to be analyzed for language identification.

    Returns:
        str: A string representing the identified language.

    Raises:
        FileNotFoundError: If the audio file cannot be loaded from the URL.
        RuntimeError: If the language identification fails.
    """
    language_id = EncoderClassifier.from_hparams(source='speechbrain/lang-id-voxlingua107-ecapa', savedir='/tmp')
    try:
        signal = language_id.load_audio(audio_url)
    except Exception as e:
        raise FileNotFoundError('The audio file could not be loaded.') from e

    try:
        prediction = language_id.classify_batch(signal)
    except Exception as e:
        raise RuntimeError('Language identification failed.') from e

    return prediction


# test_function_code --------------------

def test_identify_language_from_audio_url():
    print("Testing started.")
    # Define three different audio URLs for testing
    test_urls = [
        "https://omniglot.com/soundfiles/udhr/udhr_en.mp3",
        "https://omniglot.com/soundfiles/udhr/udhr_es.mp3",
        "https://omniglot.com/soundfiles/udhr/udhr_fr.mp3"
    ]
    expected_results = ['English', 'Spanish', 'French']

    for i, url in enumerate(test_urls):
        case_number = i + 1
        total_cases = len(test_urls)
        print(f"Testing case [{case_number}/{total_cases}] started.")
        try:
            language = identify_language_from_audio_url(url)
            assert language == expected_results[i], f"Test case [{case_number}/{total_cases}] failed: Expected {expected_results[i]}, but got {language}."
        except Exception as e:
            assert False, f"Test case [{case_number}/{total_cases}] failed with an exception: {e}."

    print("Testing finished.")


# call_test_function_line --------------------

test_identify_language_from_audio_url()