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

    Raises:
        OSError: If there is a problem with loading the model or tokenizing the input.
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
    Test the summarize_article function with some example articles.
    """
    article1 = 'Studies have shown that owning a dog is good for you.'
    article2 = 'The quick brown fox jumps over the lazy dog.'
    article3 = 'In a shocking turn of events, the cat chased the dog.'
    assert isinstance(summarize_article(article1), str)
    assert isinstance(summarize_article(article2), str)
    assert isinstance(summarize_article(article3), str)
    print('All Tests Passed')

# call_test_function_code --------------------

test_summarize_article()