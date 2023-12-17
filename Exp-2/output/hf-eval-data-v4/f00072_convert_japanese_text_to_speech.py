# requirements_file --------------------

!pip install -U transformers soundfile 

# function_import --------------------

from transformers import pipeline
import soundfile as sf

# function_code --------------------

def convert_japanese_text_to_speech(japanese_text: str, output_file: str) -> None:
    """
    Convert a given Japanese sentence into a speech audio file.

    :param japanese_text: The Japanese text to be converted to speech.
    :param output_file: The filename for the output audio file.
    """
    # Initialize the text-to-speech pipeline with the specified model
    tts = pipeline("text-to-speech", model="espnet/kan-bayashi_jvs_tts_finetune_jvs001_jsut_vits_raw_phn_jaconv_pyopenjta-truncated-178804")
    # Convert the text to speech waveform
    audio_waveform = tts(japanese_text)[0]["generated_sequence"]
    # Save the audio waveform to an audio file
    sf.write(output_file, audio_waveform, samplerate=24000)

# test_function_code --------------------

def test_convert_japanese_text_to_speech():
    print("Testing started.")
    # Example Japanese text
    japanese_text = "こんにちは、世界"
    output_file = "test_output.wav"

    # Call the function to convert text to speech
    convert_japanese_text_to_speech(japanese_text, output_file)

    # Assert that the output file is created
    assert os.path.exists(output_file), f"Test failed: {output_file} does not exist"
    print("Testing finished.")

# Run the test function
test_convert_japanese_text_to_speech()