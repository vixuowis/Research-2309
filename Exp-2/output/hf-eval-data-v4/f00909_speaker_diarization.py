# requirements_file --------------------

!pip install -U pyannote.audio==2.0

# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def speaker_diarization(audio_file_path, access_token, output_file_path):
    """
    Applies speaker diarization on an audio file and saves the output in RTTM format.

    :param audio_file_path: Path to the audio file to process.
    :param access_token: Access token for pyannote.audio pretrained model.
    :param output_file_path: Path to save the RTTM output file.
    """
    # Load the pretrained pipeline for speaker diarization
    pipeline = Pipeline.from_pretrained('pyannote/speaker-diarization@2.1', use_auth_token=access_token)

    # Process the audio file through the pipeline
    diarization = pipeline(audio_file_path)

    # Save the diarization output in RTTM format
    with open(output_file_path, 'w') as rttm_file:
        diarization.write_rttm(rttm_file)
    print(f"Diarization results saved to {output_file_path}")

# test_function_code --------------------

def test_speaker_diarization():
    print("Testing started.")
    
    audio_file_path = "test_audio.wav"  # Replace with a path to a test audio file
    access_token = "ACCESS_TOKEN_GOES_HERE"  # Use an actual access token
    output_file_path = "test_output.rttm"

    # Testing case: checking if the RTTM file is created
    print("Testing case [1/1] started.")
    speaker_diarization(audio_file_path, access_token, output_file_path)

    # Verify that the RTTM file exists after the function is called
    assert os.path.exists(output_file_path), f"Test case failed: RTTM output file {output_file_path} was not created."
    print("Testing finished.")

# To run the test, uncomment the line below
# Note: A real test would require a test audio file and an actual access token
# test_speaker_diarization()