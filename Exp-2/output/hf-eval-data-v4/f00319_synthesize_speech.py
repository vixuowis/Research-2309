# requirements_file --------------------

!pip install -U fairseq IPython

# function_import --------------------

from fairseq.checkpoint_utils import load_model_ensemble_and_task_from_hf_hub
from fairseq.models.text_to_speech.hub_interface import TTSHubInterface
import IPython.display as ipd

# function_code --------------------

def synthesize_speech(text):
    """
    Synthesizes speech from the input text using the FastSpeech2 model from the Hugging Face model hub.

    Arguments:
    text (str): The input text to be converted to speech.

    Returns:
    object: An IPython.display.Audio object that can be played in an IPython environment.
    """
    # Load the FastSpeech2 pre-trained model
    models, cfg, task = load_model_ensemble_and_task_from_hf_hub(
        'facebook/fastspeech2-en-200_speaker-cv4',
        arg_overrides={'vocoder': 'hifigan', 'fp16': False}
    )
    model = models[0]
    TTSHubInterface.update_cfg_with_data_cfg(cfg, task.data_cfg)
    generator = task.build_generator(model, cfg)

    # Prepare the sample input
    sample = TTSHubInterface.get_model_input(task, text)

    # Generate the speech wave and rate
    wav, rate = TTSHubInterface.get_prediction(task, model, generator, sample)
    # Return the audio object
    return ipd.Audio(wav, rate=rate)

# test_function_code --------------------

def test_synthesize_speech():
    print("Testing synthesize_speech function.")

    # Testing with a simple sentence
    sentence = 'Hello, world!'
    audio_output = synthesize_speech(sentence)
    assert isinstance(audio_output, ipd.Audio), "The function should return an IPython.display.Audio object."

    print("Test case passed!")

# Run the test function
test_synthesize_speech()