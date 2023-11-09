# function_import --------------------

from transformers import AutoModelForSpeechClassification, Wav2Vec2Processor
import soundfile as sf
import torch

# function_code --------------------

def identify_language(audio_file_path):
    """
    Identify the language spoken in an audio file using a pre-trained model.

    Args:
        audio_file_path (str): Path to the audio file.

    Returns:
        str: The identified language.
    """
    # Load the pre-trained model and processor
    model = AutoModelForSpeechClassification.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')
    processor = Wav2Vec2Processor.from_pretrained('sanchit-gandhi/whisper-medium-fleurs-lang-id')

    # Load the audio file
    speech, _ = sf.read(audio_file_path)

    # Preprocess the audio file
    inputs = processor(speech, sampling_rate=16_000, return_tensors='pt', padding=True)

    # Make a prediction
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

    # Get the predicted class
    predicted_class = torch.argmax(logits, dim=1)

    # Convert the predicted class to a language
    language = model.config.id2label[predicted_class.item()]

    return language

# test_function_code --------------------

def test_identify_language():
    """
    Test the identify_language function.
    """
    # Path to a test audio file
    test_audio_file_path = 'path_to_test_audio_file'

    # Call the function with the test audio file
    result = identify_language(test_audio_file_path)

    # Assert that the result is a string (the identified language)
    assert isinstance(result, str)

# call_test_function_code --------------------

test_identify_language()