# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from sentiment_analysis_model import SentimentAnalysisModel

# function_code --------------------

def transcribe_and_analyze_sentiment(audio_sample):
    """
    Transcribes an audio sample using the Whisper ASR model and analyzes the sentiment of the transcription.

    Args:
        audio_sample (dict): A dictionary containing the 'array' and 'sampling_rate' of the audio sample.

    Returns:
        str: The transcription of the audio sample.
        str: The sentiment of the transcription.
    """
    # Instantiate the model and the processor
    processor = WhisperProcessor.from_pretrained('openai/whisper-large-v2')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v2')

    # Preprocess the audio sample
    input_features = processor(audio_sample['array'], sampling_rate=audio_sample['sampling_rate'], return_tensors='pt').input_features

    # Generate predicted IDs for the transcription of the audio
    predicted_ids = model.generate(input_features)

    # Decode the predicted_ids to obtain the transcription of the audio
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    # Analyze the sentiment of the transcribed text
    sentiment_model = SentimentAnalysisModel()
    sentiment = sentiment_model.analyze_sentiment(transcription)

    return transcription, sentiment

# test_function_code --------------------

def test_transcribe_and_analyze_sentiment():
    """
    Tests the transcribe_and_analyze_sentiment function.
    """
    # Load a sample audio dataset
    ds = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample = ds[0]['audio']

    # Call the function with the sample audio
    transcription, sentiment = transcribe_and_analyze_sentiment(sample)

    # Assert that the function returns a tuple
    assert isinstance((transcription, sentiment), tuple)

    # Assert that the transcription is a string
    assert isinstance(transcription, str)

    # Assert that the sentiment is a string
    assert isinstance(sentiment, str)

# call_test_function_code --------------------

test_transcribe_and_analyze_sentiment()