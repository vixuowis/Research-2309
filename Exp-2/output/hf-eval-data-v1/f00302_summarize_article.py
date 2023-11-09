from transformers import pipeline


def summarize_article(article):
    """
    This function uses the PEGASUS model from Hugging Face Transformers to summarize a long article.
    The model is pre-trained on both C4 and HugeNews datasets and is designed to extract gap sentences and generate summaries.
    
    Parameters:
    article (str): The long article that needs to be summarized.
    
    Returns:
    str: The summarized version of the article.
    """
    # Load the PEGASUS model for summarization
    summarizer = pipeline('summarization', model='google/pegasus-xsum')
    
    # Summarize the article
    summary = summarizer(article, min_length=75, max_length=150)[0]['summary_text']
    
    return summary