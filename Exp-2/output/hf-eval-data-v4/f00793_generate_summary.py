# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_summary(article_text):
    """
    Generate a summary for a given news article using GPT-2 Large model.

    Parameters:
    - article_text (str): The text of the news article for which to generate a summary.

    Returns:
    - str: The generated summary of the news article.
    """
    # Initialize the text generation pipeline with GPT-2 Large model
    summary_generator = pipeline('text-generation', model='gpt2-large')
    
    # Generate the summary
    summary = summary_generator(article_text, max_length=50, num_return_sequences=1)[0]['generated_text']
    
    return summary

# test_function_code --------------------

def test_generate_summary():
    print("Testing generate_summary function started.")
    # Example of article text
    article_text = ("Stock prices have soared this week as several large tech companies " 
                    "reported better than expected quarterly results. This has led to a rally "
                    "in the markets amid a general atmosphere of economic recovery.")
    
    # Expected output contains the first few sentences of the given text
    expected_output_part = article_text[:90]

    # Test case 1: Generating the summary
    print("Testing case [1/1] started.")
    generated_summary = generate_summary(article_text)
    
    # The generated summary should start with the first few sentences of the given text
    assert expected_output_part in generated_summary, "Test case [1/1] failed: The generated summary does not contain the expected content."
    
    print("Test case [1/1] passed.")
    print("Testing finished.")

# Run the test function
test_generate_summary()