# requirements_file --------------------

import subprocess

requirements = ["transformers", "tokenizers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import PreTrainedTokenizerFast, BartModel

# function_code --------------------

def extract_korean_news_features(article_text):
    """
    Extracts features from a Korean news article using a pre-trained KoBART model.

    Args:
        article_text (str): A string containing the Korean news article to analyze.

    Returns:
        torch.Tensor: A tensor containing the extracted features from the news article.

    Raises:
        ValueError: If the article_text is not a string or is empty.
    """
    if not isinstance(article_text, str) or not article_text:
        raise ValueError('The article text must be a non-empty string.')

    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
    model = BartModel.from_pretrained('gogamza/kobart-base-v2')
    tokens = tokenizer(article_text, return_tensors='pt')
    features = model(**tokens)
    return features.last_hidden_state

# test_function_code --------------------

def test_extract_korean_news_features():
    print('Testing started.')

    # Test case 1: Valid input
    print('Testing case [1/3] started.')
    sample_article = '한국어 뉴스 기사 샘플입니다.'
    features = extract_korean_news_features(sample_article)
    assert isinstance(features, torch.Tensor), 'Test case [1/3] failed: Features should be a torch.Tensor'

    # Test case 2: Empty string
    print('Testing case [2/3] started.')
    try:
        extract_korean_news_features('')
        assert False, 'Test case [2/3] failed: ValueError was not raised for empty string'
    except ValueError:
        pass  # Expected

    # Test case 3: Non-string input
    print('Testing case [3/3] started.')
    try:
        extract_korean_news_features(None)
        assert False, 'Test case [3/3] failed: ValueError was not raised for non-string input'
    except ValueError:
        pass  # Expected

    print('Testing finished.')

# call_test_function_line --------------------

test_extract_korean_news_features()