from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset


def transcribe_audio(conference_audio_file):
    """
    Transcribes an audio file using the Whisper model from Hugging Face Transformers.

    Args:
        conference_audio_file (str): The path to the audio file to be transcribed.

    Returns:
        str: The transcription of the audio file.
    """
    processor = WhisperProcessor.from_pretrained('openai/whisper-small')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-small')

    audio_data = load_audio(conference_audio_file)
    input_features = processor(audio_data['array'], sampling_rate=audio_data['sampling_rate'], return_tensors='pt').input_features

    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription