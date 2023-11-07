from typing import *
from transformers import pipeline

def transcriber(url):
    '''
    Transcribes speech into text.

    :param url: The URL of the audio file to transcribe.
    :return: A dictionary containing the transcribed text.
    '''
    transcriber = pipeline(task='automatic-speech-recognition', model='openai/whisper-small')
    result = transcriber(url)
    return result
