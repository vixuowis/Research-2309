# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch

# function_code --------------------

def analyze_audio_emotion(audio_path: str) -> str:
    """
    Analyzes the emotion of the speaker in the audio file.

    Args:
        audio_path: A string path to the audio file.

    Returns:
        The predicted emotion of the speaker in the audio file.

    Raises:
        FileNotFoundError: If the audio file does not exist at the specified path.
        Exception: If there is an error processing the audio file or predicting the emotion.
    """
    # Ensure the audio file exists
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"The audio file was not found at: {audio_path}")

    # Load the pre-trained model and tokenizer
    model = Wav2Vec2ForCTC.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')
    tokenizer = Wav2Vec2Processor.from_pretrained('ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition')

    # Process the audio file for the model
    input_data = tokenizer(audio_path, return_tensors="pt")
    input_values = input_data.input_values.to("cuda")

    # Predict the emotion
    with torch.no_grad():
        predictions = model(input_values)
    predicted_ids = torch.argmax(predictions.logits, dim=-1)
    predicted_emotions = tokenizer.batch_decode(predicted_ids)

    # Return the predicted emotion
    return predicted_emotions[0]

# test_function_code --------------------

def test_analyze_audio_emotion():
    print("Testing started.")

    # Test case 1: Valid audio file
    print("Testing case [1/1] started.")
    emotion = analyze_audio_emotion('valid_audio_file.wav')
    assert isinstance(emotion, str), f"Test case [1/1] failed: The result should be a string, got {type(emotion)}"
    print("Testing case [1/1] finished.")

    print("Testing finished.")

# call_test_function_line --------------------

test_analyze_audio_emotion()