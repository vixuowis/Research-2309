# function_import --------------------

from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
import torchaudio

# function_code --------------------

def translate_speech_to_speech(audio_file_path):
    """
    Translates Hokkien speech to English speech using Fairseq's xm_transformer_s2ut_hk-en model.

    Args:
        audio_file_path (str): The path to the audio file to be translated.

    Returns:
        str: The translated English text.
    """
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_s2ut_hk-en', task='speech_to_text', cache_dir='./models')
    model = models[0].cpu()
    audio, _ = torchaudio.load(audio_file_path)
    generator = task.build_generator([model], cfg)
    sample = S2THubInterface.get_model_input(task, audio)
    translation = S2THubInterface.get_prediction(task, model, generator, sample)
    return translation

# test_function_code --------------------

def test_translate_speech_to_speech():
    """
    Tests the translate_speech_to_speech function by translating a sample Hokkien audio file and checking if the output is a string.
    """
    translation = translate_speech_to_speech('/path/to/sample/audio/file')
    assert isinstance(translation, str), 'The translation should be a string.'

# call_test_function_code --------------------

test_translate_speech_to_speech()