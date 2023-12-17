# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModel
import torch

# function_code --------------------

def analyze_russian_text_sentiment(text):
    tokenizer = AutoTokenizer.from_pretrained('sberbank-ai/sbert_large_mt_nlu_ru')
    model = AutoModel.from_pretrained('sberbank-ai/sbert_large_mt_nlu_ru')

    encoded_input = tokenizer([text], padding=True, truncation=True, max_length=24, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Assume that the sentiment analysis is done by a separate function that uses the embeddings
    sentiment = perform_sentiment_analysis(sentence_embeddings)
    return sentiment

# test_function_code --------------------

def test_analyze_russian_text_sentiment():
    print("Testing started.")
    sample_text = 'Пример текста на русском языке.'

    sentiment = analyze_russian_text_sentiment(sample_text)

    assert isinstance(sentiment, dict), f"Test case failed: Function should return a dictionary, got {type(sentiment)}"
    assert 'sentiment' in sentiment, "Test case failed: The returned dictionary should have a key 'sentiment'"
    print("Testing finished.")

test_analyze_russian_text_sentiment()