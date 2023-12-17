# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_news_article(news_article: str) -> str:
    """
    Summarizes a long news article using a pre-trained model.

    Args:
        news_article (str): The text of the news article to be summarized.

    Returns:
        str: A summary of the news article.

    Raises:
        ValueError: If the input news article is empty.
    """
    if not news_article:
        raise ValueError('The news article text cannot be empty.')

    summarizer = pipeline('summarization', model='it5/it5-base-news-summarization')
    summary = summarizer(news_article)[0]['summary_text']
    return summary

# test_function_code --------------------

def test_summarize_news_article():
    print("Testing started.")
    # Testing cases
    test_cases = [
        {
            'input': 'Dal 31 maggio Ã¨ infine partita la piattaforma ITsART, a piÃ¹ di un anno da quando... siano invece disponibili gratuitamente.',
            'expected_summary_start': 'Dal 31 maggio Ã¨ partita la piattaforma ITsART,'
        }
    ]

    for i, test in enumerate(test_cases):
        print(f"Testing case [{i+1}/{len(test_cases)}] started.")
        summary = summarize_news_article(test['input'])
        assert summary.startswith(test['expected_summary_start']), f"Test case [{i+1}/{len(test_cases)}] failed: expected summary to start with {test['expected_summary_start']} but got {summary[:len(test['expected_summary_start'])]}"
        print(f"Test case [{i+1}/{len(test_cases)}] passed.")

    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_news_article()