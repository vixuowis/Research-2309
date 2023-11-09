from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from datasets import load_dataset
import torch

# Function to convert audio to text using the Wav2Vec2 model
# The function takes an audio file as input and returns the transcribed text
# The function uses the 'facebook/wav2vec2-base-960h' model from the Transformers library
# The model has been trained on the LibriSpeech dataset

def audio_to_text(audio_file):
    # Load the pre-trained processor and model
    processor = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
    model = Wav2Vec2ForCTC.from_pretrained('facebook/wav2vec2-base-960h')

    # Load the audio file
    ds = load_dataset('patrickvonplaten/librispeech_asr_dummy', 'clean', split='validation')
    input_values = processor(ds[0]['audio']['array'], return_tensors='pt', padding='longest').input_values

    # Use the model to transcribe the audio to text
    logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    return transcription