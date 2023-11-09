from transformers import pipeline, Wav2Vec2ForCTC
import os

# Function to classify audio
# This function uses the Hugging Face Transformers library to classify audio files.
# It uses the 'anton-l/wav2vec2-random-tiny-classifier' model, which is based on wav2vec2 and is designed for audio classification tasks.
def classify_audio(audio_file_path):
    # Check if the file exists
    if not os.path.isfile(audio_file_path):
        raise ValueError(f"{audio_file_path} does not exist")
    # Create an audio classification model
    audio_classifier = pipeline('audio-classification', model=Wav2Vec2ForCTC.from_pretrained('anton-l/wav2vec2-random-tiny-classifier'))
    # Classify the audio file
    category = audio_classifier(audio_file_path)
    return category