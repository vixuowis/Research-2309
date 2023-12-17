# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import T5Tokenizer, T5Model

# function_code --------------------

def summarize_article(article_text):
    """
    Summarize the given article text using T5 large model.

    Args:
        article_text (str): The text of the article to summarize.

    Returns:
        str: The summarized text.
    """
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    model = T5Model.from_pretrained('t5-large')

    input_ids = tokenizer(f"summarize: {article_text}", return_tensors='pt').input_ids
    summary_ids = model.generate(input_ids)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# test_function_code --------------------

def test_summarize_article():
    print("Testing summarize_article function.")

    # Example article text
    article_text = "The quick brown fox jumps over the lazy dog. This is a test article to check the summarization capability of the model."

    # Expected output is not fixed, but should be a shorter version of the article
    expected_output_contains = ['quick', 'fox', 'jumps', 'dog', 'test', 'summarization']

    # Get the summarized output
    summarized_text = summarize_article(article_text)

    print(f"Summarized text: {summarized_text}")
    assert all(word in summarized_text for word in expected_output_contains), "The summarized text does not contain all expected keywords."

    print("Test passed")

# Running the test function
test_summarize_article()