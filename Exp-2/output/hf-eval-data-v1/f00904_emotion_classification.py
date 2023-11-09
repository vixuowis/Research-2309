from transformers import pipeline


def emotion_classification(audio_file_path: str, top_k: int = 1):
    """
    Function to classify emotion from an audio file using a pre-trained model.

    Args:
        audio_file_path (str): Path to the audio file.
        top_k (int, optional): Number of top predictions to return. Defaults to 1.

    Returns:
        list: List of dictionaries containing 'label' and 'score' of the top_k predictions.
    """
    emotion_classifier = pipeline('audio-classification', model='superb/wav2vec2-base-superb-er')
    emotion_label = emotion_classifier(audio_file_path, top_k=top_k)
    return emotion_label