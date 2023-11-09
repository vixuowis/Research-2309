from transformers import pipeline


def detect_voice_activity(audio_file_path: str) -> dict:
    """
    Detects voice activity in an audio file using the FSMN-VAD model from Hugging Face Transformers library.

    Args:
        audio_file_path (str): The path to the audio file to be analyzed.

    Returns:
        dict: A dictionary containing the results of the voice activity detection.
    """
    voice_activity_detector = pipeline('voice-activity-detection', model='funasr/FSMN-VAD')
    voice_activity = voice_activity_detector(audio_file_path)
    return voice_activity