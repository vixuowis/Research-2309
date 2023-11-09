from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import numpy as np
import torch
import soundfile as sf

def change_speaker_voice(audio_file, speaker_embedding_file):
    '''
    This function changes the speaker's voice in a recorded podcast using the SpeechT5 model from Hugging Face Transformers.
    Args:
    audio_file (str): Path to the audio file.
    speaker_embedding_file (str): Path to the speaker embedding file.
    Returns:
    None
    '''
    # Load the dataset
    dataset = load_dataset('hf-internal-testing/librispeech_asr_demo', 'clean', split='validation')
    example_speech = dataset[0]['audio']['array']
    sampling_rate = dataset.features['audio'].sampling_rate
    # Create instances of the processor, model, and vocoder
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_vc')
    model = SpeechT5ForSpeechToSpeech.from_pretrained('microsoft/speecht5_vc')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')
    # Create input tensors from the audio and sampling rate
    inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors='pt')
    # Load and convert the speaker embeddings into a suitable format for the model
    speaker_embeddings = np.load(speaker_embedding_file)
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
    # Generate the speech with a new speaker's voice
    speech = model.generate_speech(inputs['input_values'], speaker_embeddings, vocoder=vocoder)
    # Save the output audio to a file named 'speech.wav'
    sf.write(audio_file, speech.numpy(), samplerate=16000)