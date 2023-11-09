# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def speaker_diarization(audio_file):
    """
    This function performs speaker diarization on an audio file using the pyannote.audio library.

    Args:
        audio_file (str): The path to the audio file to be processed.

    Returns:
        Diarization: A diarization object containing information about who was speaking and when in the audio file.
    """
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1')
    diarization = pipeline(audio_file)
    return diarization

# test_function_code --------------------

def test_speaker_diarization():
    """
    This function tests the speaker_diarization function with a sample audio file.
    """
    diarization = speaker_diarization('sample_audio.wav')
    assert isinstance(diarization, type(Pipeline.from_pretrained('pyannote/speaker-diarization@2.1')('sample_audio.wav'))), 'The function should return a diarization object.'

# call_test_function_code --------------------

test_speaker_diarization()