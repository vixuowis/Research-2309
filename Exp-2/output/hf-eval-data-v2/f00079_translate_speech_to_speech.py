# function_import --------------------

from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from huggingface_hub import snapshot_download
import torchaudio
import IPython.display as ipd

# function_code --------------------

def translate_speech_to_speech(audio_path):
    """
    This function translates spoken English audio to spoken Hokkien audio using the 'facebook/xm_transformer_s2ut_en-hk' model.

    Args:
        audio_path (str): The path to the English audio file to be translated.

    Returns:
        numpy.ndarray: The translated Hokkien audio.
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
    This function tests the 'translate_speech_to_speech' function by translating a sample English audio file and checking the type of the output.
    """
    # Path to a sample English audio file
    sample_audio_path = '/path/to/sample/audio/file'

    # Translate the sample audio file
    translated_audio = translate_speech_to_speech(sample_audio_path)

    # Check the type of the output
    assert isinstance(translated_audio, numpy.ndarray), 'The output should be a numpy.ndarray.'

# call_test_function_code --------------------

test_translate_speech_to_speech()