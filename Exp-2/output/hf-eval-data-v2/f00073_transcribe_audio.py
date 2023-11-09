# function_import --------------------

from huggingsound import SpeechRecognitionModel

# function_code --------------------

def transcribe_audio(audio_paths):
    """
    Transcribe audio files into text using a pre-trained ASR model.

    Args:
        audio_paths (list): A list of paths to the audio files to be transcribed.

    Returns:
        A list of transcriptions for each audio file.
    """
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-english')
    transcriptions = model.transcribe(audio_paths)
    return transcriptions

# test_function_code --------------------

def test_transcribe_audio():
    """
    Test the transcribe_audio function with a sample audio file.

    Raises:
        AssertionError: If the transcription does not match the expected result.
    """
    audio_paths = ['path/to/sample_audio.mp3']
    expected_transcription = ['This is a sample audio file.']
    assert transcribe_audio(audio_paths) == expected_transcription

# call_test_function_code --------------------

test_transcribe_audio()