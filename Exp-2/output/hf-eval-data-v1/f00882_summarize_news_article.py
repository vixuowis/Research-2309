from transformers import PegasusForConditionalGeneration, PegasusTokenizer


def summarize_news_article(news_article: str) -> str:
    """
    Summarizes a news article using the PegasusForConditionalGeneration model from Hugging Face Transformers.

    Args:
        news_article (str): The news article to summarize.

    Returns:
        str: The summary of the news article.
    """
    model_name = 'google/pegasus-cnn_dailymail'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.encode(news_article, return_tensors='pt')
    summary_ids = model.generate(inputs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary