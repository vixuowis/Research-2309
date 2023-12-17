# requirements_file --------------------

!pip install -U fairseq huggingface_hub torchaudio os torch json

# function_import --------------------

import os
import torchaudio
from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from huggingface_hub import snapshot_download

# function_code --------------------

def translate_english_to_hokkien(audio_path):
    cache_dir = os.getenv('HUGGINGFACE_HUB_CACHE')
    models, cfg, task = hub_utils.load_model_ensemble_and_task_from_hf_hub(
        'facebook/xm_transformer_s2ut_en-hk',
        arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'},
        cache_dir=cache_dir
    )
    model = models[0].cpu()
    generator = task.build_generator([model], cfg)

    audio, _ = torchaudio.load(audio_path)
    sample = S2THubInterface.get_model_input(task, audio)
    unit = S2THubInterface.get_prediction(task, model, generator, sample)

    hkg_vocoder = snapshot_download('facebook/unit_hifigan_HK_layer12.km2500_frame_TAT-TTS', cache_dir=cache_dir)
    x = hub_utils.from_pretrained(hkg_vocoder, 'model.pt', '.', config_yaml='config.json', fp16=False, is_vocoder=True)
    vocoder_cfg = json.load(open(f"{x['args']['data']}/config.json"))
    vocoder = CodeHiFiGANVocoder(x['args']['model_path'][0], vocoder_cfg)

    wav, sr = vocoder(unit)
    return wav, sr

# test_function_code --------------------

def test_translate_english_to_hokkien():
    print("Testing started.")
    # Assuming there is a test directory with test audio files
    test_dir = './test_audio/'
    test_files = os.listdir(test_dir)
    for idx, file in enumerate(test_files, start=1):
        test_case = f"Testing case [{idx}/{len(test_files)}] started."
        try:
            print(test_case)
            wav, sr = translate_english_to_hokkien(os.path.join(test_dir, file))
            assert isinstance(wav, torch.Tensor), f"Test case [{idx}/{len(test_files)}] failed: Output 'wav' is not a tensor."
            assert isinstance(sr, int), f"Test case [{idx}/{len(test_files)}] failed: Output 'sr' is not an integer."
        except Exception as e:
            print(f"Test case [{idx}/{len(test_files)}] failed: {e}")
    print("Testing finished.")