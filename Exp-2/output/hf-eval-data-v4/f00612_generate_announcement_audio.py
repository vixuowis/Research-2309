# requirements_file --------------------

!pip install -U transformers datasets torchaudio

# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

# function_code --------------------

def generate_announcement_audio(announcement_text):
    # Load the necessary models and processor
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
    model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')

    # Prepare the input text
    inputs = processor(text=announcement_text, return_tensors='pt')

    # Load speaker embeddings to enhance speech quality
    embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]['xvector']).unsqueeze(0)

    # Generate the audio speech
    speech = model.generate_speech(inputs['input_ids'], speaker_embeddings, vocoder=vocoder)

    # Save the speech to a file
    file_name = 'announcement_speech.wav'
    sf.write(file_name, speech.numpy(), samplerate=16000)

    return file_name

# test_function_code --------------------

def test_generate_announcement_audio():
    print("Testing generate_announcement_audio function.")

    # Test case: Correct announcement
    announcement_text = "Test announcement text."
    file_name = generate_announcement_audio(announcement_text)
    assert os.path.exists(file_name), f"Test failed: {file_name} does not exist"

    print("Testing finished. All tests passed!")