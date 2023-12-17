# requirements_file --------------------

pip install transformers sentencepiece

# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_article(article_text):
    """
    Summarizes the input article text using PegasusForConditionalGeneration.

    Args:
        article_text (str): The text of the article to be summarized.

    Returns:
        str: The summarized text.
    """
    tokenizer = PegasusTokenizer.from_pretrained("tuner007/pegasus_summarizer")
    model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_summarizer")
    inputs = tokenizer.encode(article_text, return_tensors="pt", truncation=True)
    summary_ids = model.generate(inputs)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# test_function_code --------------------

def test_summarize_article():
    print("Testing started.")
    sample_text = """
    India wicket-keeper batsman Rishabh Pant has said someone from the crowd threw a ball on pacer Mohammed Siraj while he was fielding in the ongoing third Test against England on Wednesday. Pant revealed the incident made India skipper Virat Kohli upset."
    expected_summary = "Rishabh Pant reveals crowd incident upset Virat Kohli during Test against England."

    print("Testing case [1/1] started.")
    assert summarize_article(sample_text).startswith(expected_summary), f"Test case [1/1] failed: Summarized text does not match expected summary."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_article()