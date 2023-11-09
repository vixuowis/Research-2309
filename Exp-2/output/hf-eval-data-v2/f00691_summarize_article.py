# function_import --------------------

from transformers import T5Tokenizer, T5Model

# function_code --------------------

def summarize_article(article):
    """
    Summarize a lengthy article using the T5 large model from Hugging Face Transformers.

    Args:
        article (str): The article to be summarized.

    Returns:
        str: The summarized article.
    """
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    model = T5Model.from_pretrained('t5-large')
    input_ids = tokenizer('summarize: ' + article, return_tensors='pt').input_ids
    decoder_input_ids = tokenizer('summarize: ', return_tensors='pt').input_ids
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    return outputs

# test_function_code --------------------

def test_summarize_article():
    """
    Test the summarize_article function with a sample article.
    """
    article = 'Studies have shown that owning a dog is good for you.'
    summary = summarize_article(article)
    assert isinstance(summary, str), 'The output should be a string.'
    assert len(summary) < len(article), 'The summary should be shorter than the original article.'

# call_test_function_code --------------------

test_summarize_article()