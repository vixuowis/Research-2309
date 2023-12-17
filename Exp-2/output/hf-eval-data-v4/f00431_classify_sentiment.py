# requirements_file --------------------

!pip install -U transformers soundfile

# function_import --------------------

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import soundfile as sf

# function_code --------------------

def classify_sentiment(audio_file):
    """
    Classify the sentiment of Spanish audio speech.

    Args:
        audio_file (str): The path to the audio file to be analyzed.

    Returns:
        str: The label of the sentiment classified by the model, such as 'positive', 'negative' or 'neutral'.
    """
    speech, _ = sf.read(audio_file)
    processor = Wav2Vec2Processor.from_pretrained('hackathon-pln-es/wav2vec2-base-finetuned-sentiment-classification-MESD')
    inputs = processor(speech, return_tensors='pt', padding=True)
    model = Wav2Vec2ForSequenceClassification.from_pretrained('hackathon-pln-es/wav2vec2-base-finetuned-sentiment-classification-MESD')
    logits = model(**inputs).logits
    pred_ids = logits.argmax(dim=-1).item()
    label = processor.tokenizer.convert_ids_to_tokens([pred_ids])[0]
    return label

# test_function_code --------------------

def test_classify_sentiment():
    print("Testing function classify_sentiment.")
    # Assuming 'path/to/example.wav' is an existing audio file in the expected format
    example_audio_file = 'path/to/example.wav'

    # Test case 1: The function should return a string
    print("Testing case [1/1] started.")
    sentiment = classify_sentiment(example_audio_file)
    assert isinstance(sentiment, str), f"Test case [1/1] failed: Returned type {{type(sentiment)}} is not a string"
    print(sentiment)  # Optionally print the sentiment label for visual inspection
    print("Testing finished.")

# Running the test function
test_classify_sentiment()