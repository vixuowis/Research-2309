# requirements_file --------------------

!pip install -U fairseq torchaudio huggingface_hub

# function_import --------------------

from fairseq import checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
import torchaudio

# function_code --------------------

def translate_audio_hokkien_to_english(audio_file_path):
    """Translate Hokkien speech from an audio file to English text.

    Args:
        audio_file_path (str): The file path to the audio file to translate.

    Returns:
        str: The translated English text.

    Raises:
        FileNotFoundError: If the audio file does not exist.
        Exception: If translation fails.
    """
    # Load the Hokkien-to-English model
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_s2ut_hk-en', task='speech_to_text', cache_dir='./models')
    model = models[0].cpu()

    # Load the input audio file
    audio, _ = torchaudio.load(audio_file_path)

    # Prepare the model input
    generator = task.build_generator([model], cfg)
    sample = S2THubInterface.get_model_input(task, audio)

    # Generate the translated text
    translation = S2THubInterface.get_prediction(task, model, generator, sample)

    return translation

# test_function_code --------------------

def test_translate_audio_hokkien_to_english():
    print("Testing started.")
    # Assuming a hypothetical audio file for testing
    sample_audio = 'test_hokkien_audio.wav'

    # Test case 1: Check if translation is a string
    print("Testing case [1/1] started.")
    translation = translate_audio_hokkien_to_english(sample_audio)
    assert isinstance(translation, str), f"Test case [1/1] failed: Expected translation to be a string, got {type(translation)} instead."
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_audio_hokkien_to_english()