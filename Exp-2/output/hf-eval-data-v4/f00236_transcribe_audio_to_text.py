# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# function_code --------------------

def transcribe_audio_to_text(audio_path: str) -> str:
    """
    Transcribe the given audio file to text using the 'openai/whisper-large' ASR model.

    :param audio_path: The file path of the audio to be transcribed.
    :return: The transcribed text as a string.
    """
    processor = WhisperProcessor.from_pretrained('openai/whisper-large')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large')
    
    # Load the audio file
    audio_file = {'array': None, 'sampling_rate': None} # Placeholder for real audio loading method
    
    # Get input features from audio
    input_features = processor(audio_file['array'], sampling_rate=audio_file['sampling_rate'], return_tensors='pt').input_features
    
    # Generate predictions
    predicted_ids = model.generate(input_features)
    
    # Decode predicted IDs to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription


# test_function_code --------------------

def test_transcribe_audio_to_text():
    print("Testing transcribe_audio_to_text function.")
    
    # This test assumes the availability of a sample voice note, 'voice_note.wav'
    transcription = transcribe_audio_to_text('voice_note.wav')
    
    assert transcription, "The transcription of the audio file is empty."
    
    print("Test passed. Transcription: ", transcription)

# Run the test
test_transcribe_audio_to_text()