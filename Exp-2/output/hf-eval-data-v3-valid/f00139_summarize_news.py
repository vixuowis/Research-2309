# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def summarize_news(article_text: str, model_name: str = 'csebuetnlp/mT5_multilingual_XLSum') -> str:
    """
    Summarize a news article using a pre-trained model from the transformers library.

    Args:
        article_text (str): The text of the news article to be summarized.
        model_name (str, optional): The name of the pre-trained model to use. Defaults to 'csebuetnlp/mT5_multilingual_XLSum'.

    Returns:
        str: The summarized text.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer.encode(article_text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
    output_ids = model.generate(input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return summary

# test_function_code --------------------

def test_summarize_news():
    """
    Test the summarize_news function.
    """
    article_text = 'International news article text here...'
    summary = summarize_news(article_text)
    assert isinstance(summary, str), 'The output should be a string.'
    assert len(summary) > 0, 'The output should not be empty.'

    article_text = 'Another international news article text here...'
    summary = summarize_news(article_text)
    assert isinstance(summary, str), 'The output should be a string.'
    assert len(summary) > 0, 'The output should not be empty.'

    return 'All Tests Passed'

# call_test_function_code --------------------

test_summarize_news()