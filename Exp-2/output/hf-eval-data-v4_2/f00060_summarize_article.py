# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import BigBirdPegasusForConditionalGeneration, AutoTokenizer

# function_code --------------------

def summarize_article(article_text:str) -> str:
    """
    Summarize a given article using BigBird Pegasus model.

    Args:
        article_text (str): The text of the article to summarize.

    Returns:
        str: The summary of the article.

    Raises:
        ValueError: If the article_text is empty or None.
    """
    if not article_text:
        raise ValueError('The article_text must not be empty.')

    tokenizer = AutoTokenizer.from_pretrained('google/bigbird-pegasus-large-bigpatent')
    model = BigBirdPegasusForConditionalGeneration.from_pretrained('google/bigbird-pegasus-large-bigpatent')
    inputs = tokenizer(article_text, return_tensors='pt')
    prediction = model.generate(**inputs)
    summary = tokenizer.batch_decode(prediction, skip_special_tokens=True)[0]
    return summary

# test_function_code --------------------

def test_summarize_article():
    print("Testing started.")

    # Test case 1: Normal text
    print("Testing case [1/2] started.")
    article_text = 'The quick brown fox jumps over the lazy dog.'
    summary = summarize_article(article_text)
    assert summary, f"Test case [1/2] failed: Summary is empty for normal text"

    # Test case 2: Empty text
    print("Testing case [2/2] started.")
    try:
        summarize_article('')
    except ValueError as e:
        assert str(e) == 'The article_text must not be empty.', f"Test case [2/2] failed: ValueError was not raised for empty text"

    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_article()