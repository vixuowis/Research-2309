from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# Function to transcribe audio files with punctuation marks
# Uses the Hugging Face Transformers library and the 'jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli' model
# This model has been fine-tuned on the libritts and voxpopuli datasets to generate transcriptions with punctuation marks
# Suitable for transcribing podcasts

def transcribe_audio(audio_filepath):
    # Load the pre-trained model and processor
    model = Wav2Vec2ForCTC.from_pretrained('jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli')
    processor = Wav2Vec2Processor.from_pretrained('jbetker/wav2vec2-large-robust-ft-libritts-voxpopuli')

    # Preprocess the audio data and convert it to the format required by the model
    inputs = processor(audio_filepath, return_tensors="pt", padding=True)

    # Perform the transcription
    outputs = model(inputs.input_values.to("cuda"), attention_mask=inputs.attention_mask.to("cuda"), labels=inputs.labels.to("cuda"))

    # Post-process the output to obtain the final transcriptions with punctuation marks
    transcription = processor.decode(outputs.logits.argmax(dim=-1)[0])

    return transcription