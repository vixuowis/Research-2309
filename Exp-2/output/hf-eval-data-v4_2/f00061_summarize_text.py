# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def summarize_text(article_text):
    """
    Generates a summary for the given article text using a pre-trained model.

    Args:
        article_text (str): The text of the article to be summarized.

    Returns:
        str: A summary of the article.

    Raises:
        ValueError: If the article text is empty or None.
    """
    if not article_text:
        raise ValueError('The article text must not be empty.')

    model_name = 'csebuetnlp/mT5_multilingual_XLSum'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer.encode(article_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    output_ids = model.generate(input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return summary

# test_function_code --------------------

def test_summarize_text():
    print("Testing started.")
    # Simulate article text
    article_text = 'Videos that say approved vaccines are dangerous and cause autism, cancer or infertility are among those that will be taken down, the company said.'

    # Test case 1: Non-empty input
    print("Testing case [1/1] started.")
    summary = summarize_text(article_text)
    assert type(summary) == str and len(summary) > 0, f"Test case [1/1] failed: The summary should be non-empty string."
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_text()