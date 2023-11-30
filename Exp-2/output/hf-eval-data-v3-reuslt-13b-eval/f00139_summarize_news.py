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
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    model.to(torch.device('cpu'))
    
    # Encoder-decoder
    tokenized_text = tokenizer.encode("summarize: " + article_text, return_tensors="pt",  padding=True)
    output = model.generate(tokenized_text, max_length=256, length_penalty=.8)
    decoded_string = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return decoded_string

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