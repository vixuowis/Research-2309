from transformers import pipeline


def summarize_news(long_news_article):
    """
    This function uses the Hugging Face Transformers library to summarize a long news article.
    It uses the 'it5/it5-base-news-summarization' model, which is specifically designed for summarizing news articles.
    
    Args:
        long_news_article (str): The text of the long news article to be summarized.
    
    Returns:
        str: The summarized text of the news article.
    """
    # Create a summarization pipeline using the 'it5/it5-base-news-summarization' model
    summarizer = pipeline('summarization', model='it5/it5-base-news-summarization')
    # Pass the long news article text to the pipeline
    summary = summarizer(long_news_article)
    # The pipeline will return a short summary of the article
    return summary['summary_text']