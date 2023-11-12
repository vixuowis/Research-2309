# function_import --------------------

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import soundfile as sf

# function_code --------------------

model = Wav2Vec2ForSequenceClassification.from_pretrained('hackathon-pln-es/wav2vec2-base-finetuned-sentiment-classification-MESD')
processor = Wav2Vec2Processor.from_pretrained('hackathon-pln-es/wav2vec2-base-finetuned-sentiment-classification-MESD')

def classify_sentiment(audio_file):
    '''
    Classify the sentiment of the audio file using a pre-trained model.

    Args:
        audio_file (str): The path to the audio file.

    Returns:
        str: The sentiment label of the audio file.
    '''
    speech, _ = sf.read(audio_file)
    inputs = processor(speech, return_tensors='pt', padding=True)
    logits = model(**inputs).logits
    pred_ids = logits.argmax(dim=-1).item()
    label = processor.tokenizer.convert_ids_to_tokens([pred_ids])[0]
    return label

# test_function_code --------------------

def test_classify_sentiment():
    '''
    Test the classify_sentiment function.
    '''
    # Test with a positive sentiment audio file
    assert classify_sentiment('path/to/positive/audio/file.wav') == 'positive'
    # Test with a negative sentiment audio file
    assert classify_sentiment('path/to/negative/audio/file.wav') == 'negative'
    # Test with a neutral sentiment audio file
    assert classify_sentiment('path/to/neutral/audio/file.wav') == 'neutral'
    return 'All Tests Passed'

# call_test_function_code --------------------

print(test_classify_sentiment())