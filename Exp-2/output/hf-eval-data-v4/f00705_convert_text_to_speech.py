# requirements_file --------------------

!pip install -U transformers, datasets, torch, soundfile

# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

# function_code --------------------

def convert_text_to_speech(text, speaker_index=0):
    '''
    Convert text to speech using different voices from pre-trained models.

    Args:
        text (str): The text to be converted into speech.
        speaker_index (int): The index of the speaker embedding to use for voice customization.

    Returns:
        str: The path to the saved audio file.
    '''
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
    model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')

    inputs = processor(text=text, return_tensors='pt')
    embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')
    speaker_embeddings = torch.tensor(embeddings_dataset[speaker_index]['xvector']).unsqueeze(0)

    speech = model.generate_speech(inputs['input_ids'], speaker_embeddings, vocoder=vocoder)
    audio_file_path = 'speech.wav'
    sf.write(audio_file_path, speech.numpy(), samplerate=16000)
    return audio_file_path

# test_function_code --------------------

def test_convert_text_to_speech():
    print("Testing started.")
    test_text = 'Hello, how are you?'
    output_path = convert_text_to_speech(test_text, speaker_index=7306)

    print("Testing case [1/1] started.")
    assert os.path.exists(output_path), f"Test case [1/1] failed: Audio file not found at {output_path}"
    assert os.path.getsize(output_path) > 0, f"Test case [1/1] failed: Audio file at {output_path} is empty"
    print("Testing finished.")