from transformers import PreTrainedTokenizerFast, BartModel


def extract_features_from_korean_news(news_article):
    """
    This function extracts features from a Korean news article using the 'gogamza/kobart-base-v2' model.
    
    Parameters:
    news_article (str): The Korean news article from which to extract features.
    
    Returns:
    torch.Tensor: The extracted features from the news article.
    """
    # Initialize the tokenizer and model
    tokenizer = PreTrainedTokenizerFast.from_pretrained('gogamza/kobart-base-v2')
    model = BartModel.from_pretrained('gogamza/kobart-base-v2')
    
    # Tokenize the news article
    tokens = tokenizer(news_article, return_tensors='pt')
    
    # Extract features from the news article
    features = model(**tokens)
    
    return features