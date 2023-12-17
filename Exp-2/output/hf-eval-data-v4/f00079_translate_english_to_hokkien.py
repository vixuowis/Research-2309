# requirements_file --------------------

!pip install -U torchaudio fairseq huggingface_hub 

# function_import --------------------

import torchaudio
from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from huggingface_hub import snapshot_download


# function_code --------------------

def translate_english_to_hokkien(audio_path):
    cache_dir = snapshot_download('facebook/xm_transformer_s2ut_en-hk')
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task_from_hf_hub(
        'facebook/xm_transformer_s2ut_en-hk',
        arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'},
        cache_dir=cache_dir
    )
    model = models[0].cpu()
    cfg['task'].cpu = True

    generator = task.build_generator([model], cfg)

    audio, _ = torchaudio.load(audio_path)
    sample = S2THubInterface.get_model_input(task, audio)
    hokkien_translation = S2THubInterface.get_prediction(task, model, generator, sample)

    return hokkien_translation


# test_function_code --------------------

def test_translate_english_to_hokkien():
    print('Testing started.')
    audio_path = 'path/to/english/audio/sample.wav'

    print('Translating English Audio to Hokkien...')
    translation = translate_english_to_hokkien(audio_path)
    assert translation is not None, 'Translation failed.'

    print('Testing case [1/1] passed.')
    print('Testing finished.')

# Run the test function
test_translate_english_to_hokkien()
