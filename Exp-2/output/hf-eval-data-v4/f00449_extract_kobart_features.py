# requirements_file --------------------

!pip install -U transformers==latest tokenizers==latest

# function_import --------------------

from transformers import PreTrainedTokenizerFast, BartModel

# function_code --------------------

def extract_kobart_features(news_article: str):
    """
    This function takes a Korean news article as input and uses the pre-trained KoBART model
    to extract features from it.
    
    Parameters:
    news_article (str): A string containing the Korean news article text.
    
    Returns:
    Tensor: A tensor of feature vectors extracted from the news article using the KoBART model.
    """
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
    model = BartModel.from_pretrained('gogamza/kobart-base-v2')
    
    tokens = tokenizer(news_article, return_tensors='pt')
    outputs = model(**tokens)
    
    return outputs.last_hidden_state

# test_function_code --------------------

def test_extract_kobart_features():
    print("Testing started.")
    
    # Test data: Sample Korean news article
    sample_news_article = "이것은 한국어 뉴스 기사의 예제입니다. 코바트 모델을 사용하여 특징 추출을 수행합니다."
    
    # Testing feature extraction using a sample Korean news article
    print("Testing extraction case [1/1] started.")
    features = extract_kobart_features(sample_news_article)
    
    assert features is not None, "Test case failed: The function did not return any features."
    assert features.size(0) == 1, "Test case failed: The number of examples should be 1."
    assert features.ndim == 3, "Test case failed: The features tensor should have 3 dimensions."
    
    print("Testing finished.")

# Run the test function
test_extract_kobart_features()