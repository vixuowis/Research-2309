# requirements_file --------------------

import subprocess

requirements = ["transformers", "datasets", "torch", "soundfile"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf


# function_code --------------------

def generate_audio_announcement(text, speaker_id=7306, output_file='speech.wav'):
  """Generate an audio speech from text using SpeechT5 model.

  Args:
    text (str): The text to be converted to speech.
    speaker_id (int, optional): The speaker ID for speaker embeddings. Defaults to 7306.
    output_file (str, optional): The output file to save the speech audio. Defaults to 'speech.wav'.

  Returns:
    str: The path to the generated speech audio file.

  Raises:
    ValueError: If the text input is None or empty.
    RuntimeError: If any model or dataset loading fails.
  """
  if not text:
    raise ValueError('The text input cannot be None or empty.')
  try:
    processor = SpeechT5Processor.from_pretrained('microsoft/speecht5_tts')
    model = SpeechT5ForTextToSpeech.from_pretrained('microsoft/speecht5_tts')
    vocoder = SpeechT5HifiGan.from_pretrained('microsoft/speecht5_hifigan')
    inputs = processor(text=text, return_tensors='pt')
    embeddings_dataset = load_dataset('Matthijs/cmu-arctic-xvectors', split='validation')
    speaker_embeddings = torch.tensor(embeddings_dataset[speaker_id]['xvector']).unsqueeze(0)
    speech = model.generate_speech(inputs['input_ids'], speaker_embeddings, vocoder=vocoder)
    sf.write(output_file, speech.numpy(), samplerate=16000)
    return output_file
  except Exception as e:
    raise RuntimeError(f'An error occurred while generating the audio announcement: {e}')


# test_function_code --------------------

def test_generate_audio_announcement():
  print("Testing started.")

  # Test case 1: Valid input
  print("Testing case [1/2] started.")
  expected_output = 'speech.wav'
  result = generate_audio_announcement("Hello world")
  assert result == expected_output, f"Test case [1/2] failed: Expected {expected_output}, got {result}"

  # Test case 2: Empty text input
  print("Testing case [2/2] started.")
  try:
    generate_audio_announcement("")
    assert False, "Test case [2/2] failed: ValueError not raised for empty text input"
  except ValueError:
    pass  # Expected

  print("Testing finished.")


# call_test_function_line --------------------

test_generate_audio_announcement()