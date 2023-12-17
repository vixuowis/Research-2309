# requirements_file --------------------

!pip install -U pyannote.audio>=2.0

# function_import --------------------

from pyannote.audio import Pipeline

# function_code --------------------

def generate_transcript_with_timestamps(audio_file_path, model_path='pyannote/speaker-diarization@2.1', output_path='transcript.rttm', use_auth_token=None):
    """
    Generate a transcript with timestamps and speaker labels for a given audio file using the pyannote.audio speaker diarization model.

    Parameters:
    audio_file_path (str): Path to the audio file for which to generate the transcript.
    model_path (str): The path or identifier of the pre-trained model. Default is 'pyannote/speaker-diarization@2.1'.
    output_path (str): Path to save the generated transcript file in RTTM format.
    use_auth_token (str): The token to use for authentication, if required by the model. Default is None.

    Returns:
    str: The path to the saved transcript file.
    """
    # Initialize the diarization pipeline with the provided model
    pipeline = Pipeline.from_pretrained(model_path, use_auth_token=use_auth_token)

    # Process the audio file with the pipeline
    diarization = pipeline(audio_file_path)

    # Write diarization result to file
    with open(output_path, 'w') as file:
        diarization.write_rttm(file)

    return output_path

# test_function_code --------------------

def test_generate_transcript_with_timestamps():
    print("Testing function: generate_transcript_with_timestamps.")

    # A hypothetical audio file path for testing
    sample_audio = 'path/to/sample_audio.wav'

    # Expected output path
    expected_output = 'transcript.rttm'

    # Test the function with a default model
    print("Testing with default model.")
    output_file = generate_transcript_with_timestamps(sample_audio)
    assert os.path.isfile(output_file), f"Test failed: output file {output_file} does not exist."

    # Test with a custom model (assuming it is available)
    custom_model_path = 'path/to/custom_model'

    print("Testing with custom model.")
    output_file_custom = generate_transcript_with_timestamps(sample_audio, model_path=custom_model_path)
    assert os.path.isfile(output_file_custom), f"Test failed: output file {output_file_custom} does not exist."

    print("All tests passed.")