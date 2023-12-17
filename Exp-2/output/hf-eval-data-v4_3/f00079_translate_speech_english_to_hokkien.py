# requirements_file --------------------

import subprocess

requirements = ["fairseq", "huggingface_hub", "torchaudio", "IPython"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from huggingface_hub import snapshot_download
import torchaudio
import IPython.display as ipd

# function_code --------------------

def translate_speech_english_to_hokkien(audio_path):
    """Translate spoken English audio to spoken Hokkien audio.

    Args:
        audio_path (str): The file path to the input English audio.

    Returns:
        IPython.display.Audio: An object that can be used to play the translated audio in Jupyter notebooks.

    Raises:
        FileNotFoundError: If the audio file is not found at the given path.
        RuntimeError: If the translation model could not process the input audio.
    """

    # Load model
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task_from_hf_hub(
        'facebook/xm_transformer_s2ut_en-hk',
        arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'}
    )
    model = models[0].cpu()

    # Load audio
    audio, _ = torchaudio.load(audio_path)

    # Generate translated speech
    sample = S2THubInterface.get_model_input(task, audio)
    hokkien_translation = S2THubInterface.get_prediction(task, model, None, sample)

    # Play translated audio
    return ipd.Audio(hokkien_translation, rate=16000)  # Assuming 16 kHz sample rate

# test_function_code --------------------

def test_translate_speech_english_to_hokkien():
    print("Testing started.")

    # Test case 1: Check if FileNotFoundError is raised for a non-existent file
    print("Testing case [1/2] started.")
    try:
        translate_speech_english_to_hokkien('non_existent_file.wav')
    except FileNotFoundError:
        assert True
    else:
        assert False, "Test case [1/2] failed: FileNotFoundError was not raised."

    # Test case 2: Check if the function returns an Audio object for a valid file
    print("Testing case [2/2] started.")
    # Assuming 'sample_english.wav' exists and is a valid audio file
    result = translate_speech_english_to_hokkien('sample_english.wav')
    assert isinstance(result, ipd.Audio), "Test case [2/2] failed: The function did not return an Audio object."
    print("Testing finished.")

# call_test_function_line --------------------

test_translate_speech_english_to_hokkien()