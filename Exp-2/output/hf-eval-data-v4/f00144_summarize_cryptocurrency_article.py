# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

import transformers

# function_code --------------------

def summarize_cryptocurrency_article(article_text):
    """
    Summarize the given article about cryptocurrency investment risks.

    :param article_text: str - the text of the article to be summarized
    :return: str - the summary of the article
    """
    # Initialize the text-generation pipeline with the LLaMA-13B model
    generator = transformers.pipeline('text-generation', model='decapoda-research/llama-13b-hf')
    # Use the model to generate a summary
    summary = generator(article_text, max_length=100, do_sample=False)
    # Return the generated summary text
    return summary[0]['generated_text']

# test_function_code --------------------

def test_summarize_cryptocurrency_article():
    print("Testing summary generation started.")

    # The article text to summarize
    article_text = 'Cryptocurrencies have become exceedingly popular among investors seeking higher returns and diversification in their portfolios. However, investing in these digital currencies carries several inherent risks. These include market volatility, difficulty in predicting future values, potential frauds and insecurity due to lack of regulation, and environmental concerns surrounding digital currency mining.'

    # Expected summary (this is a hypothetical summary for testing purposes)
    expected_summary = 'Cryptocurrency investments offer higher returns but come with risks like market volatility, regulatory issues, and environmental concerns.'

    # Generate the summary
    summary = summarize_cryptocurrency_article(article_text)

    # Compare the generated summary with the expected summary
    assert summary == expected_summary, f"Test failed: Expected summary does not match actual.\nExpected: {expected_summary}\nActual: {summary}"

    print("Testing summary generation finished.")

# Run the test function
test_summarize_cryptocurrency_article()