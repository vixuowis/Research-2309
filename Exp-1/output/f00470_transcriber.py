from typing import *
from transformers import pipeline

def transcriber(audio_file):
    # This function uses a finetuned model for automatic speech recognition to transcribe the given audio file.
    # Args:
    #     audio_file (str): The path to the audio file.
    # Returns:
    #     dict: A dictionary containing the transcribed text.
    transcriber = pipeline("automatic-speech-recognition", model="stevhliu/my_awesome_asr_minds_model")
    return transcriber(audio_file)
