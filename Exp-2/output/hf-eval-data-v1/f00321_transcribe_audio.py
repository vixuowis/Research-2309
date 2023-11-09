from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
import torch


def transcribe_audio(audio_data):
    '''
    This function transcribes audio data into text using the pre-trained model 'openai/whisper-tiny.en' from Hugging Face Transformers.
    
    Parameters:
    audio_data (np.array): The audio data to be transcribed.
    
    Returns:
    transcription (str): The transcribed text.
    '''
    # Load the processor and model
    processor = WhisperProcessor.from_pretrained('openai/whisper-tiny.en')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny.en')
    
    # Process the audio data and generate the transcription
    input_features = processor(audio_data, sampling_rate=16000, return_tensors='pt').input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    
    return transcription