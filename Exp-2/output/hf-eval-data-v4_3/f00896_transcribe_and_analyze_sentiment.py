# requirements_file --------------------

import subprocess

requirements = ["transformers", "datasets"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset
from sentiment_model import SentimentAnalyzer

# function_code --------------------

def transcribe_and_analyze_sentiment(audio_file_path):
    """Transcribes an audio file using Whisper ASR model and analyzes the sentiment.

    Args:
        audio_file_path (str): The file path of the audio file to transcribe and analyze.

    Returns:
        tuple: A pair containing the transcription (str) and sentiment analysis result.

    Raises:
        FileNotFoundError: If the audio file does not exist at the specified path.
    """
    # Ensure the audio file exists
    if not os.path.exists(audio_file_path):
        raise FileNotFoundError(f'Audio file not found at {audio_file_path}')

    # Initialize Whisper components
    processor = WhisperProcessor.from_pretrained('openai/whisper-large-v2')
    model = WhisperForConditionalGeneration.from_pretrained('openai/whisper-large-v2')

    # Load the audio file
    sample = processor.load_audio(audio_file_path)
    input_features = processor(sample['array'], sampling_rate=sample['sampling_rate'], return_tensors='pt').input_features

    # Transcribe the audio
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    # Analyze the sentiment
    sentiment_analyzer = SentimentAnalyzer()
    sentiment_result = sentiment_analyzer.analyze_sentiment(transcription)

    return transcription, sentiment_result

# test_function_code --------------------

def test_transcribe_and_analyze_sentiment():
    print("Testing started.")
    # Assuming load_dataset functionality to get a dummy audio file path
    dataset = load_dataset('hf-internal-testing/librispeech_asr_dummy', 'clean', split='validation')
    sample_data = dataset[0]  # Using sample data from the dataset

    # Test case 1: Valid audio file
    print("Testing case [1/2] started.")
    transcription, sentiment = transcribe_and_analyze_sentiment(sample_data['file'])
    assert isinstance(transcription, str), f"Test case [1/2] failed: Expected transcription as string, got {type(transcription)}"
    assert 'sentiment' in sentiment, f"Test case [1/2] failed: Sentiment analysis result should have 'sentiment' key"

    # Test case 2: Non-existent audio file
    print("Testing case [2/2] started.")
    try:
        transcribe_and_analyze_sentiment('non_existent_file.wav')
        assert False, "Test case [2/2] failed: Expected FileNotFoundError for non-existent file"
    except FileNotFoundError:
        pass  # expected

    print("Testing finished.")

# call_test_function_line --------------------

test_transcribe_and_analyze_sentiment()