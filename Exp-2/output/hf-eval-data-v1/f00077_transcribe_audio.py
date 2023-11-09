from huggingsound import SpeechRecognitionModel
import torch

# Function to transcribe audio files into Chinese text
# Uses the Hugging Face Transformers library and a pretrained model fine-tuned for Chinese speech recognition
# @param audio_paths: List of paths to audio files to be transcribed
# @return: List of transcriptions

def transcribe_audio(audio_paths):
    # Load the pretrained model
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn')
    # Transcribe the audio files
    transcriptions = model.transcribe(audio_paths)
    return transcriptions