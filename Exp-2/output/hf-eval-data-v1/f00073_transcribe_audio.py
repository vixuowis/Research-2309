from huggingsound import SpeechRecognitionModel
import torch

# Function to transcribe audio files into text
# Uses the Hugging Face Transformers library and a pre-trained model
# The model is fine-tuned on a large-scale English dataset
# The function takes a list of paths to audio files as input
# Returns a list of transcriptions

def transcribe_audio(audio_paths):
    # Create an instance of the SpeechRecognitionModel
    model = SpeechRecognitionModel('jonatasgrosman/wav2vec2-large-xlsr-53-english')
    # Transcribe the audio files
    transcriptions = model.transcribe(audio_paths)
    return transcriptions