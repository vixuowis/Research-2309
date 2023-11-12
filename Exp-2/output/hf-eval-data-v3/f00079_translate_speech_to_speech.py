# function_import --------------------

from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from huggingface_hub import snapshot_download
import torchaudio
import numpy

# function_code --------------------

def translate_speech_to_speech(audio_path):
    """
    Translates spoken English audio to spoken Hokkien audio using the 'facebook/xm_transformer_s2ut_en-hk' model.

    Args:
        audio_path (str): Path to the English audio file to be translated.

    Returns:
        numpy.ndarray: Translated Hokkien audio.
    """
    # Load model
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_s2ut_en-hk', arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'})
    model = models[0].cpu()

    # Load audio
    audio, _ = torchaudio.load(audio_path)

    # Generate translated speech
    sample = S2THubInterface.get_model_input(task, audio)
    hokkien_translation = S2THubInterface.get_prediction(task, model, generator, sample)

    return hokkien_translation

# test_function_code --------------------

def test_translate_speech_to_speech():
    """
    Tests the translate_speech_to_speech function with a sample audio file.
    """
    # Test with a sample audio file
    audio_path = 'sample_audio.wav'
    hokkien_translation = translate_speech_to_speech(audio_path)

    assert isinstance(hokkien_translation, numpy.ndarray), 'The translated audio should be a numpy array.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_translate_speech_to_speech()