# requirements_file --------------------

!pip install -U fairseq torchaudio huggingface_hub

# function_import --------------------

from fairseq import hub_utils, checkpoint_utils
from fairseq.models.speech_to_text.hub_interface import S2THubInterface
import torchaudio

# function_code --------------------

def translate_hokkien_to_english_speech(audio_file_path: str) -> str:
    """
    Translates spoken Hokkien in an audio file to English text using Fairseq's S2UT model.

    :param audio_file_path: Path to the input audio file containing Hokkien speech
    :return: Translated English text
    """
    # Load the Hokkien-to-English model and task
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task_from_hf_hub(
        'facebook/xm_transformer_s2ut_hk-en',
        task='speech_to_text',
        cache_dir='./models'
    )
    model = models[0].cpu()
    generator = task.build_generator([model], cfg)

    # Load the audio file
    audio, _ = torchaudio.load(audio_file_path)
    
    # Prepare the model input
    sample = S2THubInterface.get_model_input(task, audio)
    
    # Generate translated text
    translation = S2THubInterface.get_prediction(task, model, generator, sample)

    return translation

# test_function_code --------------------

def test_translate_hokkien_to_english_speech():
    print("Testing translate_hokkien_to_english_speech function.")

    # You should replace 'path/to/audio_test_file_hokkien.wav' with a real Hokkien audio file
    audio_file_path = 'path/to/audio_test_file_hokkien.wav'

    # Test case: Translate Hokkien speech to English
    translation = translate_hokkien_to_english_speech(audio_file_path)
    print("Translation:", translation)
    assert isinstance(translation, str), f"Translation should be a string, but got {type(translation)}"
    
    print("Testing completed successfully.")

# Run the test function
test_translate_hokkien_to_english_speech()