# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def perform_speaker_diarization(audio_file: str, output_file: str = 'output_audio.rttm'):
    """
    Perform speaker diarization on an audio file using a pre-trained model from pyannote.audio.

    Args:
        audio_file (str): Path to the audio file.
        output_file (str, optional): Path to the output file in RTTM format. Defaults to 'output_audio.rttm'.

    Returns:
        None. Writes the diarization result to the output file.
    """
    diarization_pipeline = Pipeline.from_pretrained('philschmid/pyannote-speaker-diarization-endpoint')
    diarization = diarization_pipeline(audio_file)
    with open(output_file, 'w') as rttm:
        diarization.write_rttm(rttm)

# test_function_code --------------------

def test_perform_speaker_diarization():
    """
    Test the perform_speaker_diarization function.
    """
    # Use a sample audio file for testing
    audio_file = 'sample_audio.wav'
    output_file = 'test_output.rttm'
    perform_speaker_diarization(audio_file, output_file)
    # Check if the output file is created
    assert os.path.exists(output_file)
    # Check if the output file is not empty
    assert os.path.getsize(output_file) > 0

# call_test_function_code --------------------

test_perform_speaker_diarization()