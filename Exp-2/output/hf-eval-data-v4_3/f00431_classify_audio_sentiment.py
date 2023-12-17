# requirements_file --------------------

import subprocess

requirements = ["transformers", "soundfile"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import soundfile as sf

# function_code --------------------

def classify_audio_sentiment(audio_file: str) -> str:
    """
    Classifies the sentiment of given Spanish audio file.

    Args:
        audio_file: The path to a .wav file containing Spanish speech.

    Returns:
        The sentiment label ('POSITIVE', 'NEGATIVE', or 'NEUTRAL') inferred from the audio.

    Raises:
        ValueError: If the audio file is not found or is in an incorrect format.
    """
    try:
        speech, _ = sf.read(audio_file)
        inputs = processor(speech, return_tensors='pt', padding=True)
        logits = model(**inputs).logits
        pred_ids = logits.argmax(dim=-1).item()
        label = processor.tokenizer.convert_ids_to_tokens([pred_ids])[0]
    except Exception as e:
        raise ValueError(str(e))
    return label

# test_function_code --------------------

def test_classify_audio_sentiment():
    print("Testing started.")
    # Assume 'sample.wav' is a valid audio file with Spanish speech
    sample_audio_file = 'sample.wav'

    # Test case 1:
    print("Testing case [1/1] started.")
    sentiment = classify_audio_sentiment(sample_audio_file)
    assert sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL'], f"Test case [1/1] failed: Unexpected sentiment {sentiment}"
    print("Testing finished.")

# call_test_function_line --------------------

test_classify_audio_sentiment()