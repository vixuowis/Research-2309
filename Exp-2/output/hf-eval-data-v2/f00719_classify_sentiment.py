# function_import --------------------

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# function_code --------------------

def classify_sentiment(audio_file):
    """
    Classify the sentiment of a Spanish audio file using a pre-trained model.

    Args:
        audio_file (str): Path to the audio file to be analyzed.

    Returns:
        str: The classified sentiment ('positive', 'neutral', 'negative').

    Raises:
        Exception: If the audio file cannot be processed or the sentiment cannot be classified.
    """
    try:
        # Load the pre-trained model
        model = Wav2Vec2ForSequenceClassification.from_pretrained('hackathon-pln-es/wav2vec2-base-finetuned-sentiment-classification-MESD')

        # Load the audio file and process it
        processor = Wav2Vec2Processor.from_pretrained('hackathon-pln-es/wav2vec2-base-finetuned-sentiment-classification-MESD')
        input_values = processor(audio_file, return_tensors='pt').input_values

        # Classify the sentiment
        logits = model(input_values).logits
        predicted_ids = torch.argmax(logits, dim=-1)

        # Convert the predicted IDs to sentiment labels
        sentiment_labels = ['negative', 'neutral', 'positive']
        return sentiment_labels[predicted_ids]
    except Exception as e:
        raise Exception('Failed to classify sentiment: ' + str(e))

# test_function_code --------------------

def test_classify_sentiment():
    """
    Test the classify_sentiment function by classifying the sentiment of a sample audio file.
    """
    # Define a sample audio file
    sample_audio_file = 'sample.wav'

    # Classify the sentiment of the sample audio file
    sentiment = classify_sentiment(sample_audio_file)

    # Assert that the sentiment is one of the expected values
    assert sentiment in ['negative', 'neutral', 'positive'], 'Unexpected sentiment: ' + sentiment

# call_test_function_code --------------------

test_classify_sentiment()