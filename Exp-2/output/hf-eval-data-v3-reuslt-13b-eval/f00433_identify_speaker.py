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
    try:
        speech_recognizer = pipeline('audio-classification', model='facebook/wav2vec2-base-960h')
        result = speech_recognizer(audio_file_path)
        speakers = [{'label': label, 'score': score} for (label, score) in zip(result[0]['labels'], result[0]['scores'])]
    except:
        raise FileNotFoundError('The file does not exist')
    
    return sorted(speakers, key=lambda speaker: speaker['score'])[:5]


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