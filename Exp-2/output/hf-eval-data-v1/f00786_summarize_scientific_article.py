from transformers import pipeline


def summarize_scientific_article(article: str) -> str:
    """
    Summarize a scientific article using the 'google/pegasus-large' model from Hugging Face Transformers.

    Args:
        article (str): The text of the scientific article to be summarized.

    Returns:
        str: The summary of the article.
    """
    summarizer = pipeline('summarization', model='google/pegasus-large')
    summary = summarizer(article)
    return summary