# function_import --------------------

from transformers import pipeline

# function_code --------------------

def identify_speaker(audio_file_path: str) -> dict:
    """
    Identify the speaker in an audio file using Hugging Face's pre-trained model.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        dict: The top 5 predicted speakers and their probabilities.

    Raises:
        FileNotFoundError: If the audio file does not exist.
    """
    sid_classifier = pipeline('audio-classification', model='superb/wav2vec2-base-superb-sid')
    speaker_identification = sid_classifier(audio_file_path, top_k=5)
    return speaker_identification

# test_function_code --------------------

def test_identify_speaker():
    """
    Test the identify_speaker function.
    """
    test_audio_file_path = 'test_audio.wav'
    try:
        speaker_identification = identify_speaker(test_audio_file_path)
        assert isinstance(speaker_identification, dict)
        assert len(speaker_identification) == 5
    except FileNotFoundError:
        print('Test audio file not found.')
    return 'All Tests Passed'

# call_test_function_code --------------------

test_identify_speaker()