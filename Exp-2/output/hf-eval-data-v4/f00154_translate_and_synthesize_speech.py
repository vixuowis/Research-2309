# requirements_file --------------------

!pip install -U fairseq hub_utils torchaudio IPython.display

# function_import --------------------



# function_code --------------------


from fairseq import hub_utils
from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
from fairseq.models.text_to_speech import CodeHiFiGANVocoder
from fairseq.models.text_to_speech.hub_interface import TTSInterface
import torchaudio
import IPython.display as ipd


def translate_and_synthesize_speech(sentence, target_lang_code):
    model_id = 'facebook/unit_hifigan_mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj_dur'

    # Load the model ensemble and task
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(model_id)
    model = models[0].cpu()
    cfg['task'].cpu = True
    generator = task.build_generator([model], cfg)

    # Translate the sentence to the target language
    translation_result = task.translate(sentence, target_lang_code=target_lang_code)
    translated_sentence = translation_result['sentence']

    # Synthesize speech from the translated sentence
    vocoder_path = hub_utils.from_pretrained(model_id)['args']['vocoder_path']
    vocoder_cfg = hub_utils.from_pretrained(vocoder_path)['args']['vocoder_cfg']
    vocoder = CodeHiFiGANVocoder.from_pretrained(vocoder_path, vocoder_cfg)
    tts_model = TTSInterface(vocoder_cfg, vocoder)

    # Creating audio waveform
    audio_waveform, sample_rate = tts_model.synthesize_waveform(translated_sentence)

    return ipd.Audio(audio_waveform, rate=sample_rate)


# test_function_code --------------------


from fairseq import hub_utils
from fairseq.models.text_to_speech.hub_interface import TTSInterface

def test_translate_and_synthesize_speech():
    print("Testing started.")

    # Testing with a sample English sentence
    sentence = 'Hello world'
    target_lang_code = 'fr'  # French code

    audio = translate_and_synthesize_speech(sentence, target_lang_code)

    # Test if the audio is an instance of IPython.display.Audio
    print("Testing case [1/1] started.")
    assert isinstance(audio, ipd.Audio), f"Test case [1/1] failed: Expected IPython.display.Audio, got {type(audio)}"
    print("Testing case [1/1] passed.")

    print("Testing finished.")

# Run the test function
test_translate_and_synthesize_speech()
