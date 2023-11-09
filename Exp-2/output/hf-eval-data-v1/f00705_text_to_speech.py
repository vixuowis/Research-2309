from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

# Function to convert text to speech
# text: The text to be converted to speech
# speaker_id: The id of the speaker whose voice is to be used
# Returns: The path of the audio file containing the speech

def text_to_speech(text: str, speaker_id: int) -> str:
    # Load the necessary models
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
    model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')

    # Process the input text
    inputs = processor(text=text, return_tensors='pt')

    # Load the speaker embeddings
    embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')
    speaker_embeddings = torch.tensor(embeddings_dataset[speaker_id]['xvector']).unsqueeze(0)

    # Generate the speech
    speech = model.generate_speech(inputs['input_ids'], speaker_embeddings, vocoder=vocoder)

    # Save the speech as an audio file
    audio_file_path = 'speech.wav'
    sf.write(audio_file_path, speech.numpy(), samplerate=16000)

    return audio_file_path