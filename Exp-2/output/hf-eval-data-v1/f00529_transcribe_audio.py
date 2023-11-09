from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa


def transcribe_audio(audio_file_path):
    '''
    Transcribe an audio file using the Whisper ASR model from Hugging Face Transformers.

    Parameters:
    audio_file_path (str): Path to the audio file to transcribe.

    Returns:
    str: The transcription of the audio file.
    '''
    # Load the Whisper ASR model and processor
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')

    # Load the audio file
    audio_data, sampling_rate = librosa.load(audio_file_path, sr=None)

    # Process the audio file and prepare it for the model
    input_features = processor(audio_data, sampling_rate=sampling_rate, return_tensors='pt').input_features

    # Generate the transcription
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription