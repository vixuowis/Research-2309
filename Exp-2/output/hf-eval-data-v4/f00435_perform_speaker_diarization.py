# requirements_file --------------------

!pip install -U pyannote.audio

# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def perform_speaker_diarization(audio_file_path, output_path="output.rttm"):
    """
    Perform speaker diarization on the input audio file using pyannote.audio.
    
    Parameters:
        audio_file_path (str): Path to the input audio file.
        output_path (str): Path to save the RTTM output file.

    Returns:
        None
    """
    # Load the pre-trained speaker diarization model
    diarization_pipeline = Pipeline.from_pretrained('philschmid/pyannote-speaker-diarization-endpoint')

    # Perform diarization on the input audio file
    diarization = diarization_pipeline(audio_file_path)

    # Write the diarization result to an RTTM file
    with open(output_path, 'w') as rttm_file:
        diarization.write_rttm(rttm_file)

# test_function_code --------------------

def test_perform_speaker_diarization():
    print("Testing perform_speaker_diarization function.")
    # Assuming 'example.wav' is a valid audio file used for testing
    audio_file_path = 'example.wav'
    output_path = 'test_output.rttm'

    # Call the function with the test audio file
    perform_speaker_diarization(audio_file_path, output_path)

    # Check if the RTTM file has been created
    assert os.path.exists(output_path), f"RTTM output file not created."

    # Here more tests can be implemented (e.g. RTTM content validation)
    print("Testing completed.")

# Run the test function
test_perform_speaker_diarization()