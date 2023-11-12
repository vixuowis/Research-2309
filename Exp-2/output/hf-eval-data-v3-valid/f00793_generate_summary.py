# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_summary(article_text: str, max_length: int = 50, num_return_sequences: int = 1) -> str:
    """
    Generate a brief summary for a given article using GPT-2 Large model.

    Args:
        article_text (str): The first few sentences of the news article.
        max_length (int, optional): The maximum length of the generated summary. Defaults to 50.
        num_return_sequences (int, optional): The number of return sequences. Defaults to 1.

    Returns:
        str: The generated summary of the news article.
    """
    summary_generator = pipeline('text-generation', model='gpt2-large')
    summary = summary_generator(article_text, max_length=max_length, num_return_sequences=num_return_sequences)[0]['generated_text']
    return summary

# test_function_code --------------------

def test_generate_summary():
    """
    Test the generate_summary function.
    """
    article_text = "The first few sentences of the news article go here..."
    summary = generate_summary(article_text)
    assert isinstance(summary, str), 'The result is not a string.'
    assert len(summary) <= 50, 'The length of the summary is more than 50.'
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_summary()