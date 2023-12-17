# requirements_file --------------------

!pip install -U transformers datasets

# function_import --------------------

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

# function_code --------------------

def predict_sentiment(audio_path):
    """
    Predict the sentiment of Spanish speech in the given audio file.

    Args:
        audio_path (str): The path to the audio file to be analyzed.

    Returns:
        str: The predicted sentiment (positive, neutral, or negative).

    Raises:
        FileNotFoundError: If the audio file is not found at the specified path.
        Exception: If the prediction cannot be made due to unexpected issues.
    """
    try:
        model_name = 'hackathon-pln-es/wav2vec2-base-finetuned-sentiment-classification-MESD'
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        processor = Wav2Vec2Processor.from_pretrained(model_name)

        audio_input, sampling_rate = processor(audio_path, return_tensors='pt', sampling_rate=16000)
        output = model(**audio_input)
        sentiment = output.logits.argmax(-1).item()

        sentiment_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
        return sentiment_mapping.get(sentiment, 'unknown')
    except FileNotFoundError:
        raise FileNotFoundError(f'Audio file not found: {audio_path}')
    except Exception as e:
        raise Exception(f'Could not predict sentiment: {e}')

# test_function_code --------------------

def test_predict_sentiment():
    print("Testing started.")
    # Let's assume there are three mock audio files for the sentiment analysis
    mock_audio_files = ['mock_audio_positive.wav', 'mock_audio_neutral.wav', 'mock_audio_negative.wav']
    expected_results = ['positive', 'neutral', 'negative']

    for i, (audio_file, expected) in enumerate(zip(mock_audio_files, expected_results), 1):
        print(f"Testing case [{i}/3] started.")
        sentiment = predict_sentiment(audio_file)
        assert sentiment == expected, f"Test case [{i}/3] failed: Expected {{expected}}, got {{sentiment}}"
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_sentiment()