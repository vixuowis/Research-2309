# requirements_file --------------------

!pip install -U fairseq IPython

# function_import --------------------

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

# function_code --------------------

def text_to_speech(text):
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        'facebook/fastspeech2-en-200_speaker-cv4',
        arg_overrides={'vocoder': 'hifigan', 'fp16': False}
    )
    model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)
    sample = TTSHubInterface.get_model_input(task, text)
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    return ipd.Audio(wav, rate=rate)

# test_function_code --------------------

def test_text_to_speech():
    print('Testing text_to_speech function.')
    text = 'Hello, this is a test run.'
    audio_output = text_to_speech(text)
    assert isinstance(audio_output, ipd.Audio), 'The output must be an IPython.display.Audio object.'
    print('text_to_speech function works correctly.')

test_text_to_speech()