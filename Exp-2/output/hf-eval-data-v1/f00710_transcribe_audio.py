from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import numpy as np

# Function to transcribe audio using the Whisper model
# @param audio_file_path: Path to the audio file to be transcribed
# @return: Transcription of the audio file

def transcribe_audio(audio_file_path):
    # Load the Whisper processor and model
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')

    # Load the audio file
    sample_audio_file = audio_file_path
    sample = {'array': np.random.rand(16000), 'sampling_rate': 16000}

    # Preprocess the audio file and generate input features
    input_features = processor(sample['array'], sampling_rate=sample['sampling_rate'], return_tensors='pt').input_features

    # Generate a prediction
    predicted_ids = model.generate(input_features)

    # Decode the prediction to a transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription