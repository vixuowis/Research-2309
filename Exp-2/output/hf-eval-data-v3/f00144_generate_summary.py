# function_import --------------------

from transformers import pipeline

# function_code --------------------

def generate_summary(text: str) -> str:
    '''
    Generate a short summary of a given text using the LLaMA-13B model from Hugging Face Transformers.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The generated summary of the input text.
    '''
    summarizer = pipeline('text-generation', model='decapoda-research/llama-13b-hf')
    summary = summarizer(text)
    return summary

# test_function_code --------------------

def test_generate_summary():
    '''
    Test the function generate_summary.
    '''
    text1 = 'Cryptocurrencies have become exceedingly popular among investors seeking higher returns and diversification in their portfolios. However, investing in these digital currencies carries several inherent risks.'
    text2 = 'The environmental impact of mining digital currencies like Bitcoin has come under scrutiny, questioning the long-term sustainability of the cryptocurrency market.'
    assert isinstance(generate_summary(text1), str)
    assert isinstance(generate_summary(text2), str)
    return 'All Tests Passed'

# call_test_function_code --------------------

test_generate_summary()