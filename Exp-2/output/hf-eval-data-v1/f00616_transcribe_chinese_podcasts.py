from huggingsound import SpeechRecognitionModel
import torch

# Function to transcribe Chinese podcasts
# This function uses the Hugging Face Transformers library and a pre-trained model for Chinese speech recognition
# The model is 'jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn'
# The function takes a list of paths to audio files as input and returns a list of transcriptions

def transcribe_chinese_podcasts(audio_paths):
    # Load the pre-trained model
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')
    # Transcribe the audio files
    transcriptions = model.transcribe(audio_paths)
    return transcriptions