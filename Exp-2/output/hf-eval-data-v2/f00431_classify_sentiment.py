# function_import --------------------

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import soundfile as sf

# function_code --------------------

model = Wav2Vec2ForSequenceClassification.from_pretrained('hackathon-pln-es/wav2vec2-base-finetuned-sentiment-classification-MESD')
processor = Wav2Vec2Processor.from_pretrained('hackathon-pln-es/wav2vec2-base-finetuned-sentiment-classification-MESD')

def classify_sentiment(audio_file):
    """
    Classify the sentiment of a given audio file using a pre-trained model.

    Args:
        audio_file (str): The path to the audio file to be classified.

    Returns:
        str: The classified sentiment of the audio file (e.g., 'positive', 'negative', 'neutral').
    """
    speech, _ = sf.read(audio_file)
    inputs = processor(speech, return_tensors='pt', padding=True)
    logits = model(**inputs).logits
    pred_ids = logits.argmax(dim=-1).item()
    label = processor.tokenizer.convert_ids_to_tokens([pred_ids])[0]
    return label

# test_function_code --------------------

def test_classify_sentiment():
    """
    Test the classify_sentiment function with a sample audio file.
    """
    sentiment = classify_sentiment('path/to/sample/audio/file.wav')
    assert isinstance(sentiment, str), 'The output should be a string.'
    assert sentiment in ['positive', 'negative', 'neutral'], 'The output should be either positive, negative, or neutral.'

# call_test_function_code --------------------

test_classify_sentiment()