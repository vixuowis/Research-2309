# requirements_file --------------------

!pip install -U fairseq, torchaudio, huggingface_hub

# function_import --------------------

import json
import os
import torchaudio
import IPython.display as ipd
from fairseq import hub_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from huggingface_hub import snapshot_download

# function_code --------------------

def hokkien_to_english_translation(audio_file_path):
    """
    Translates spoken language from Hokkien to English.
    :param audio_file_path: The path to the audio file in Hokkien.
    :return: The translated audio in English.
    """
    # Load pre-trained models and configurations
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub('facebook/xm_transformer_s2ut_hk-en', arg_overrides={'config_yaml': 'config.yaml', 'task': 'speech_to_text'})
    model = models[0].cpu()
    cfg['task'].cpu = True
    generator = task.build_generator([model], cfg)

    # Load audio file
    audio, _ = torchaudio.load(audio_file_path)

    # Translate Hokkien audio to English
    sample = S2THubInterface.get_model_input(task, audio)
    translation = S2THubInterface.get_prediction(task, model, generator, sample)

    # Initialize Vocoder for text-to-speech
    cache_dir = snapshot_download('facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur')
    x = hub_utils.from_pretrained(cache_dir, 'model.pt', '.', archive_map=CodeHiFiGANVocoder.hub_models(), config_yaml='config.json', fp16=False, is_vocoder=True)
    with open(os.path.join(x['args']['data'], 'config.json')) as f:
        vocoder_cfg = json.load(f)
    vocoder = CodeHiFiGANVocoder(x['args']['model_path'][0], vocoder_cfg)
    tts_model = VocoderHubInterface(vocoder_cfg, vocoder)

    # Convert translated text to speech
    tts_sample = tts_model.get_model_input(translation)
    wav, sr = tts_model.get_prediction(tts_sample)

    # Return the audio in English
    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_hokkien_to_english_translation():
    print("Testing hokkien_to_english_translation function.")

    # Assume we have a test audio file in Hokkien
    test_audio_file = 'test_hokkien.wav'

    # Run the translation function
    result = hokkien_to_english_translation(test_audio_file)

    # Check if the result is not None
    assert result is not None, "Translation failed: result is None"
    # Check if the result is an instance of IPython.display.Audio
    assert isinstance(result, ipd.Audio), "The result should be an IPython.display.Audio instance"

    print("Testing completed successfully.")

# Execute the test function
test_hokkien_to_english_translation()