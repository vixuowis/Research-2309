from transformers import pipeline


def detect_voice_segments(audio_file_path):
    """
    This function uses the Hugging Face's pipeline function to load the 'Eklavya/ZFF_VAD' model
    which is a Voice Activity Detection (VAD) model. It then uses this model to detect voice segments
    in the provided audio file.

    Parameters:
    audio_file_path (str): The path to the audio file to be analyzed.

    Returns:
    list: A list of voice segments detected in the audio file.
    """

    # Load the voice activity detection model
    vad = pipeline('voice-activity-detection', model='Eklavya/ZFF_VAD')

    # Analyze the recording to detect voice segments
    voice_segments = vad(audio_file_path)

    return voice_segments