# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def summarize_news_article(article_text: str, model_name: str = 'csebuetnlp/mT5_multilingual_XLSum') -> str:
    """
    Summarize a news article using a pre-trained model from the transformers package.

    Args:
        article_text (str): The text of the news article to be summarized.
        model_name (str, optional): The name of the pre-trained model to use for summarization. Defaults to 'csebuetnlp/mT5_multilingual_XLSum'.

    Returns:
        str: The summarized text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer(article_text, return_tensors='pt', padding=True, truncation=True, max_length=512).input_ids
    output_ids = model.generate(input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)[0]
    summary = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return summary

# test_function_code --------------------

def test_summarize_news_article():
    """
    Test the summarize_news_article function.
    """
    article_text = 'This is a test news article. It contains some information that needs to be summarized.'
    expected_summary = 'Test news article needs summarization.'
    assert summarize_news_article(article_text) == expected_summary

    article_text = 'Another test news article. This one is a bit longer and contains more information.'
    expected_summary = 'Longer test article with more information.'
    assert summarize_news_article(article_text) == expected_summary

    article_text = 'A third test news article. This one is even longer and contains even more information.'
    expected_summary = 'Even longer article with more information.'
    assert summarize_news_article(article_text) == expected_summary

    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_news_article()