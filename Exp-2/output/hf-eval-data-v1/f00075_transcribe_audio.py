from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset


def transcribe_audio(audio_sample):
    """
    This function transcribes the given audio sample using the openai/whisper-tiny model.
    
    Parameters:
    audio_sample (dict): A dictionary containing the 'array' and 'sampling_rate' of the audio sample.
    
    Returns:
    str: The transcribed text.
    """
    # Load the WhisperProcessor and the model
    processor = WhisperProcessor.from_pretrained('openai/whisper-tiny')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-tiny')
    
    # Process the raw audio data
    input_features = processor(audio_sample['array'], sampling_rate=audio_sample['sampling_rate'], return_tensors='pt').input_features
    
    # Use the model to generate a transcription
    predicted_ids = model.generate(input_features)
    
    # Decode the transcription and return the result
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]