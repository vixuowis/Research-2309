from pyannote.audio import Pipeline
import os


def analyze_conference_call(audio_file_path: str) -> dict:
    """
    Analyze a conference call recording to identify the speakers and the segments of the conversation they participated in.

    Args:
        audio_file_path (str): The path to the audio file to be analyzed.

    Returns:
        dict: A dictionary containing the diarization results.
    """
    # Ensure the audio file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f"The audio file {audio_file_path} does not exist.")

    # Load the pre-trained model
    pipeline = Pipeline.from_pretrained('philschmid/pyannote-speaker-diarization-endpoint')

    # Analyze the audio file
    diarization = pipeline(audio_file_path)

    return diarization