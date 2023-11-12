# function_import --------------------

from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
from huggingface_hub import snapshot_download
import torchaudio
import IPython.display as ipd

# function_code --------------------

def convert_speech_to_speech(input_audio_file):
    """
    Convert English speech into different language speech.

    Args:
        input_audio_file (str): Path to the input audio file in English.

    Returns:
        IPython.lib.display.Audio: The translated audio in Spanish.
    """
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task_from_hf_hub('facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur')
    model = models[0].cpu()
    cfg['task'].cpu = True
    generator = task.build_generator([model], cfg)

    audio, _ = torchaudio.load(input_audio_file)
    sample = S2THubInterface.get_model_input(task, audio)
    translation_unit = S2THubInterface.get_prediction(task, model, generator, sample)

    cache_dir = None
    cache_dir = snapshot_download('facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur', cache_dir=cache_dir)
    x = hub_utils.from_pretrained(cache_dir, 'model.pt', '.', archive_map=CodeHiFiGANVocoder.hub_models(), config_yaml='config.json', fp16=False, is_vocoder=True)

    vocoder = CodeHiFiGANVocoder(x['args']['model_path'][0], x['model_cfg'])
    tts_model = VocoderHubInterface(x['model_cfg'], vocoder)
    tts_sample = tts_model.get_model_input(translation_unit)
    wav, sr = tts_model.get_prediction(tts_sample)
    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_convert_speech_to_speech():
    """
    Test the function convert_speech_to_speech.
    """
    # Test with a sample audio file
    result = convert_speech_to_speech('sample_audio.flac')
    assert isinstance(result, ipd.lib.display.Audio), 'The result is not an instance of IPython.lib.display.Audio'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_convert_speech_to_speech()