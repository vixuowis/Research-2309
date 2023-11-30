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
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cpu')

    input = "summarize: " + article_text
    encoding = tokenizer(input, return_tensors='pt')
    
    summary = model.generate(encoding['input_ids'], length_penalty=2.0, max_length=142, min_length=56, 
                             num_beams=4, early_stopping=True)
    
    return tokenizer.decode(summary[0], skip_special_tokens=True)

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