# requirements_file --------------------

!pip install -U speechbrain torchaudio

# function_import --------------------

from speechbrain.pretrained import EncoderClassifier, load_audio

# function_code --------------------

def detect_spoken_languages(audio_file_path):
    """
    Detect the languages spoken in an audio recording of an international conference call.

    Parameters:
        audio_file_path (str): The file path to the audio recording of the conference call.

    Returns:
        list: A list of predicted languages spoken in the audio recording.
    """
    # Initialize the language identification model
    language_id = EncoderClassifier.from_hparams(source='speechbrain/lang-id-voxlingua107-ecapa', savedir='/tmp')

    # Load the audio file
    signal = load_audio(audio_file_path)

    # Predict the spoken language
    prediction = language_id.classify_batch(signal)

    # Return the identified languages
    return prediction

# test_function_code --------------------

def test_detect_spoken_languages():
    print("Testing started.")
    # Assuming 'conference_call_example.mp3' is a valid audio file in the test environment
    test_audio_path = 'conference_call_example.mp3'

    # Testing the detect_spoken_languages function
    print("Testing detect_spoken_languages function.")
    detected_languages = detect_spoken_languages(test_audio_path)
    assert isinstance(detected_languages, list), "The function should return a list"
    assert len(detected_languages) > 0, "The function should identify at least one language"

    # Print the results of the detection
    print("Detected languages:", detected_languages)
    print("Testing finished.")

# Run the test function
test_detect_spoken_languages()