from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# Function to transcribe audio using the Whisper ASR model
# @param audio_data: The audio data to be transcribed
# @param audio_sampling_rate: The sampling rate of the audio data
# @return: The transcription of the audio data

def transcribe_audio(audio_data, audio_sampling_rate):
    # Initialize the WhisperProcessor and the WhisperForConditionalGeneration model
    processor = WhisperProcessor.from_pretrained('openai/whisper-large-v2')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v2')

    # Process the audio data to generate input features
    input_features = processor(audio_data, sampling_rate=audio_sampling_rate, return_tensors='pt').input_features

    # Use the Whisper ASR model to generate the predicted_ids from the input_features
    predicted_ids = model.generate(input_features)

    # Decode the predicted_ids to obtain the transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription