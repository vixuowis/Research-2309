from typing import *
from transformers import pipeline

def audio_classification(audio_url):
    """Classifies the audio from the given URL and returns a list of predictions.

    :param audio_url: The URL of the audio file to classify.
    :return: A list of dictionaries containing the prediction score and label."""
    classifier = pipeline(task="audio-classification", model="superb/hubert-base-superb-er")
    preds = classifier(audio_url)
    preds = [{"score": round(pred["score"], 4), "label": pred["label"]} for pred in preds]
    return preds
