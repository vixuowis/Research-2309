# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_article(article_text):
    """
    Summarize the given article text related to cryptocurrency investment risks.

    Args:
        article_text (str): A string containing the full text of the article to be summarized.

    Returns:
        str: A string containing the summarized version of the provided article.

    """
    summary_pipeline = pipeline('text-generation', model='decapoda-research/llama-13b-hf')
    summary = summary_pipeline(article_text, max_length=100, min_length=30, do_sample=False)[0]['generated_text']
    return summary

# test_function_code --------------------

def test_summarize_article():
    print("Testing started.")
    article_text = 'Cryptocurrencies have become exceedingly popular among investors seeking higher returns and diversification in their portfolios. However, investing in these digital currencies carries several inherent risks. Market volatility is a major factor – cryptocurrencies can experience wild price swings, sometimes even within hours or minutes. This high volatility makes it difficult to predict the future value of the investments and can result in significant losses. Furthermore, the lack of regulatory oversight and security concerns may also lead to potential frauds and hacks, exposing investors to additional risk. Lastly, the environmental impact of mining digital currencies like Bitcoin has come under scrutiny, questioning the long-term sustainability of the cryptocurrency market.'

    # Test case 1: Check if the function returns a summarized text
    print("Testing case [1/1] started.")
    summary = summarize_article(article_text)
    assert isinstance(summary, str) and len(summary) <= 100, f"Test case [1/1] failed: The summary is not a valid string within the expected length."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_article()