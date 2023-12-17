# requirements_file --------------------

import subprocess

requirements = ["huggingface_hub", "fairseq"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from huggingface_hub import unit
from fairseq import TTS

# function_code --------------------

def convert_book_to_audio(book_text):
    """
    Converts the text content of a book into an audio file using Text-to-Speech 
    technology.

    Args:
        book_text (str): The text content of the book to be converted to audio.

    Returns:
        str: The file path to the generated audio file.
    """
    # Load the pre-trained Text-to-Speech model
    model = unit.TTS.from_pretrained('facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10')
    
    # Convert the book text to an audio waveform
    waveform = model.generate_audio(book_text)
    
    # Save the audio waveform to a file
    audio_file_path = 'audiobook_output.wav'
    waveform.save(audio_file_path)
    
    return audio_file_path

# test_function_code --------------------

def test_convert_book_to_audio():
    print("Testing started.")

    # Test case 1: Check if the output is a non-empty string.
    print("Testing case [1/2] started.")
    book_text = "The sample book content goes here."
    audio_file_path = convert_book_to_audio(book_text)
    assert isinstance(audio_file_path, str) and audio_file_path, "Test case [1/2] failed: The returned value should be a non-empty string representing the file path."

    # Test case 2: Check if the output file path ends with .wav.
    print("Testing case [2/2] started.")
    assert audio_file_path.endswith('.wav'), "Test case [2/2] failed: The file path should end with .wav."
    print("Testing finished.")

# call_test_function_line --------------------

test_convert_book_to_audio()