from typing import *
from gtts import gTTS
from IPython.display import Audio
def narrate_dutch_text(text):
    # Convert text to speech
    tts = gTTS(text, lang='nl')
    # Save audio file
    audio_file = 'output.mp3'
    tts.save(audio_file)
    # Display audio
    Audio(audio_file)
