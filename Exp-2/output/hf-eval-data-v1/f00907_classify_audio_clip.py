from transformers import pipeline


def classify_audio_clip(audio_clip_path: str) -> str:
    """
    Classify the audio clip to determine whether it is silent or contains speech.

    Args:
        audio_clip_path (str): The path to the audio clip to be classified.

    Returns:
        str: The classification result, indicating whether the audio clip contains speech or is silent.

    Raises:
        Exception: If the audio clip path is not valid or the audio clip cannot be processed.
    """
    try:
        # Create a voice activity detection model
        vad_model = pipeline('voice-activity-detection', model='Eklavya/ZFF_VAD')
        # Classify the audio clip
        classification_result = vad_model(audio_clip_path)
        return classification_result
    except Exception as e:
        print(f'Error: {e}')
        raise