from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# Function to transcribe audio using the Whisper model
# @param audio_file: The audio file to transcribe
# @return: The transcription of the audio file
def transcribe_audio(audio_file):
    # Load the Whisper model and processor
    processor = WhisperProcessor.from_pretrained('openai/whisper-medium')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-medium')
    model.config.forced_decoder_ids = None

    # Load the audio file
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample = ds[0][audio_file]

    # Process the audio file and generate the transcription
    input_features = processor(sample['array'], sampling_rate=sample['sampling_rate'], return_tensors='pt').input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription