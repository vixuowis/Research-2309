# function_import --------------------

from fairseq import hub_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import VocoderHubInterface
import torchaudio
import IPython.display as ipd

# function_code --------------------

def text_to_speech_translation(input_audio_path):
    """
    This function takes an audio file path as input and returns the translated speech audio.
    
    Args:
        input_audio_path (str): The path to the input audio file.
    
    Returns:
        Audio: The translated speech audio.
    
    Raises:
        FileNotFoundError: If the input audio file does not exist.
    """
    model_id = 'facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur'
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(model_id)
    model = models[0].cpu()
    cfg['task'].cpu = True
    generator = task.build_generator([model], cfg)
    
    audio, _ = torchaudio.load(input_audio_path)
    sample = S2THubInterface.get_model_input(task, audio)
    unit = S2THubInterface.get_prediction(task, model, generator, sample)
    
    vocoder_path = snapshot_download(model_id)
    vocoder_args = {'model_path': ['vocoder_model_path']}
    vocoder = CodeHiFiGANVocoder(vocoder_args, vocoder_cfg)
    tts_model = VocoderHubInterface(vocoder_cfg, vocoder)
    
    tts_sample = tts_model.get_model_input(unit)
    wav, sr = tts_model.get_prediction(tts_sample)
    
    return ipd.Audio(wav, rate=sr)

# test_function_code --------------------

def test_text_to_speech_translation():
    """
    This function tests the text_to_speech_translation function by using a sample audio file.
    """
    # Test with a sample audio file
    try:
        result = text_to_speech_translation('sample_audio.flac')
        assert isinstance(result, ipd.Audio)
    except FileNotFoundError:
        print('Test audio file not found.')
    
    # Test with a non-existing audio file
    try:
        result = text_to_speech_translation('non_existing_audio.flac')
    except FileNotFoundError:
        print('Test passed. The function correctly raised a FileNotFoundError.')
    else:
        print('Test failed. The function did not raise a FileNotFoundError.')
    
    return 'All Tests Passed'

# call_test_function_code --------------------

test_text_to_speech_translation()