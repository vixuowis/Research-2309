# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import BartTokenizer, BartForConditionalGeneration

# function_code --------------------

def summarize_article(article_text):
    """
    Summarize a long article using the pre-trained BART model.

    :param article_text: str - The article text to be summarized
    :return: str - The summary of the article
    """
    model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
    tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')
    inputs = tokenizer(article_text, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=50, early_stopping=True)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

# test_function_code --------------------

def test_summarize_article():
    print("Testing article summarization.")
    # Sample long article text
    article_text = "In today's fast-paced world, where the amount of information is overwhelming, a summarization tool can be incredibly useful to quickly understand the main points of a long article. This text serves as an example to test the summarization capability of the BART model."

    # Expected summary (or the gist of it)
    expected_summary = "A tool that summarizes long articles assists in quickly understanding the main points."

    # Generate summary
    summary = summarize_article(article_text)

    # Test assertion
    assert expected_summary in summary, f"Summary does not match expected output. Generated summary: {summary}"
    print("Article summarization test passed.")

# Run the test
test_summarize_article()