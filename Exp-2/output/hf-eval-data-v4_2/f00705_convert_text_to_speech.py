# requirements_file --------------------

!pip install -U transformers datasets torch soundfile

# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

# function_code --------------------

def convert_text_to_speech(email_message, voice_index=7306):
    """Converts text to speech using SpeechT5 models.

    Args:
        email_message (str): The email message to be read aloud.
        voice_index (int, optional): The index of the speaker voice embedding from the dataset.
            Defaults to 7306.

    Returns:
        str: The path to the saved audio file.

    Raises:
        ValueError: If the email_message is empty.
    """
    # Check for empty message
    if not email_message:
        raise ValueError('The email_message cannot be empty.')

    # Initialize the SpeechT5Processor and models
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
    model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')

    # Process the input text to obtain tensors
    inputs = processor(text=email_message, return_tensors='pt')

    # Load speaker embeddings
    embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')
    speaker_embeddings = torch.tensor(embeddings_dataset[voice_index]['xvector']).unsqueeze(0)

    # Generate the speech signal
    speech = model.generate_speech(inputs['input_ids'], speaker_embeddings, vocoder=vocoder)

    # Save the speech as a .wav file
    audio_file_path = 'speech.wav'
    sf.write(audio_file_path, speech.numpy(), samplerate=16000)

    return audio_file_path


# test_function_code --------------------

def test_convert_text_to_speech():
    print("Testing started.")

    # Available dataset for testing
    embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')

    # Testing case 1: Regular usage with a specified message
    print("Testing case [1/2] started.")
    try:
        audio_path = convert_text_to_speech("Your email message here")
        assert audio_path == 'speech.wav', f"Test case [1/2] failed: Expected 'speech.wav', got {audio_path}"
    except Exception as e:
        assert False, f"Test case [1/2] failed with exception: {e}"

    # Testing case 2: Passing an empty message
    print("Testing case [2/2] started.")
    try:
        convert_text_to_speech("")
        assert False, "Test case [2/2] failed: Empty email_message did not raise ValueError"
    except ValueError as e:
        assert str(e) == 'The email_message cannot be empty.', f"Test case [2/2] failed: {e}"
    print("Testing finished.")


# call_test_function_line --------------------

test_convert_text_to_speech()