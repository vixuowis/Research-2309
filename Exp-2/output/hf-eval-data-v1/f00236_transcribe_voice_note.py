from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# Function to transcribe voice notes using the Whisper model
# @param audio: The audio file to be transcribed
# @param sampling_rate: The sampling rate of the audio file
# @return: The transcribed text

def transcribe_voice_note(audio, sampling_rate):
    # Load the pre-trained 'openai/whisper-large' model and processor
    processor = WhisperProcessor.from_pretrained('openai/whisper-large')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large')
    model.config.forced_decoder_ids = None

    # Convert the audio input into input features suitable for the model
    input_features = processor(audio, sampling_rate=sampling_rate, return_tensors='pt').input_features

    # Generate predicted IDs using the Whisper model
    predicted_ids = model.generate(input_features)

    # Decode the predicted IDs into text transcription
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

    return transcription