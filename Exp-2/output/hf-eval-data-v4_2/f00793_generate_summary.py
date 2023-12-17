# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_summary(article_text):
    """
    Generate a summary for a given piece of news article text.

    Args:
        article_text (str): The text of the news article.

    Returns:
        str: The generated summary of the article.

    Raises:
        ValueError: If article_text is not provided.

    """
    if not article_text:
        raise ValueError('No article text provided for summary generation.')
    summary_generator = pipeline('text-generation', model='gpt2-large')
    summary = summary_generator(article_text, max_length=50, num_return_sequences=1)[0]['generated_text']
    return summary

# test_function_code --------------------

def test_generate_summary():
    print("Testing started.")

    # Text case 1: Check if the function returns a string
    print("Testing case [1/1] started.")
    summary = generate_summary("The economy has been seeing an unprecedented growth this quarter...")
    assert isinstance(summary, str), f"Test case [1/1] failed: Expected a string, got {type(summary)}"
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_summary()