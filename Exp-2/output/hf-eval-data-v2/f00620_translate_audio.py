# function_import --------------------

from fairseq.models.wav2vec.wav2vec2_asr import Wav2Vec2Model
from huggingface_hub import cached_download
import soundfile as sf

# function_code --------------------

def translate_audio(input_audio_path: str, model_path: str = 'https://huggingface.co/facebook/textless_sm_cs_en/resolve/main/model.pt') -> str:
    """
    Translates a Czech audio file to English using a pretrained model.

    Args:
        input_audio_path (str): The path to the Czech language audio file.
        model_path (str, optional): The URL of the pretrained model. Defaults to 'https://huggingface.co/facebook/textless_sm_cs_en/resolve/main/model.pt'.

    Returns:
        str: The path to the translated English audio file.
    """
    # Download and load the pretrained model
    model = Wav2Vec2Model.from_pretrained(cached_download(model_path))

    # Translate the audio
    english_audio = model.translate(input_audio_path)

    # Save the translated audio to a file
    output_audio_path = 'translated_audio.wav'
    sf.write(output_audio_path, english_audio, 16000)

    return output_audio_path

# test_function_code --------------------

def test_translate_audio():
    """
    Tests the translate_audio function.
    """
    # Path to a sample Czech audio file
    input_audio_path = 'sample_czech_audio.wav'

    # Call the function with the sample audio file
    output_audio_path = translate_audio(input_audio_path)

    # Check that the output is a string
    assert isinstance(output_audio_path, str)

    # Check that the output file exists
    assert os.path.exists(output_audio_path)

# call_test_function_code --------------------

test_translate_audio()