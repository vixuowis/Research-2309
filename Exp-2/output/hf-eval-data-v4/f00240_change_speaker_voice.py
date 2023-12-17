# requirements_file --------------------

!pip install -U transformers datasets numpy torch soundfile

# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import numpy as np
import torch
import soundfile as sf

# function_code --------------------

def change_speaker_voice(audio, sampling_rate, speaker_embedding_path, output_path='transformed_speech.wav'):
    """
    Changes the speaker's voice in an audio recording using a pretrained SpeechT5 model.

    Parameters:
        audio (np.array): The input audio array.
        sampling_rate (int): The sampling rate of the audio.
        speaker_embedding_path (str): Path to the .npy file containing speaker embeddings.
        output_path (str): Path to save the transformed speech.
    """
    # Load the pretrained models and processor
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_vc')
    model = SpeechT5ForSpeechToSpeech.from_pretrained('microsoft/speecht5_vc')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')

    # Process the input audio
    inputs = processor(audio=audio, sampling_rate=sampling_rate, return_tensors='pt')

    # Load speaker embeddings
    speaker_embeddings = np.load(speaker_embedding_path)
    speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)

    # Generate the speech with the new speaker's voice
    transformed_speech = model.generate_speech(inputs['input_values'], speaker_embeddings, vocoder=vocoder)

    # Save the transformed speech to a file
    sf.write(output_path, transformed_speech.numpy(), samplerate=sampling_rate)

    return output_path

# test_function_code --------------------

def test_change_speaker_voice():
    print("Testing change_speaker_voice function started.")
    # Load a sample dataset
    dataset = load_dataset('hf-internal-testing/librispeech_asr_demo', 'clean', split='validation')
    example_speech = dataset[0]['audio']['array']
    sampling_rate = dataset.features['audio'].sampling_rate

    # Test case: Change voice of a sample audio
    print("Testing case [1/1] started.")
    transformed_path = change_speaker_voice(example_speech, sampling_rate, 'xvector_speaker_embedding.npy')
    assert os.path.isfile(transformed_path), f"Test case [1/1] failed: Output file not created."
    print("Testing finished.")

# Run the test function
if __name__ == '__main__':
    test_change_speaker_voice()