# function_import --------------------

from transformers import BarthezTokenizer, BarthezModel

# function_code --------------------

def summarize_french_news(news_article_french):
    """
    Summarize a French news article using a pre-trained model from Hugging Face Transformers.

    Args:
        news_article_french (str): The French news article to be summarized.

    Returns:
        str: The summary of the news article.
    """
    tokenizer = BarthezTokenizer.from_pretrained('moussaKam/barthez-orangesum-abstract')
    model = BarthezModel.from_pretrained('moussaKam/barthez-orangesum-abstract')
    inputs = tokenizer(news_article_french, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids=inputs["input_ids"])
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_summarize_french_news():
    """
    Test the summarize_french_news function.
    """
    news_article_french = "L'article de presse en français ici..."
    summary = summarize_french_news(news_article_french)
    assert isinstance(summary, str), "The function should return a string."
    assert len(summary) > 0, "The function should return a non-empty string."

# call_test_function_code --------------------

test_summarize_french_news()