# requirements_file --------------------

!pip install -U huggingface_hub, fairseq

# function_import --------------------

from huggingface_hub import from_pretrained
from fairseq import TTS

# function_code --------------------

def convert_text_to_audio(text, model_name='facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_es_css10', output_file='audiobook_output.wav'):
    """
    Convert the given text to an audio file using a pre-trained Text-to-Speech model.

    Parameters:
        text (str): Text content to be converted to speech.
        model_name (str): Name of the pre-trained TTS model. Default is a multilingual model.
        output_file (str): Name of the output audio file. Defaults to 'audiobook_output.wav'.

    Returns:
        None
    """
    # Load the pre-trained TTS model
    model = TTS.from_pretrained(model_name)

    # Generate audio waveform
    waveform = model.generate_audio(text)

    # Save the generated waveform as an audio file
    waveform.save(output_file)

# test_function_code --------------------

def test_convert_text_to_audio():
    print("Testing convert_text_to_audio function.")

    # Test case 1: Convert a simple text to audio
    print("Testing case [1/1] started.")
    convert_text_to_audio('Hello, this is a test.', output_file='test_output.wav')
    assert os.path.exists('test_output.wav'), "Test case [1/1] failed: The audio file was not created."
    os.remove('test_output.wav') # Clean up test file
    print("Testing case [1/1] finished.")

# Run the test function
test_convert_text_to_audio()