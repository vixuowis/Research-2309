# requirements_file --------------------

!pip install -U transformers sentencepiece

# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_article(article_text):
    # Create a tokenizer and model instance for the Pegasus summarizer
    tokenizer = PegasusTokenizer.from_pretrained("tuner007/pegasus_summarizer")
    model = PegasusForConditionalGeneration.from_pretrained("tuner007/pegasus_summarizer")

    # Tokenize the long article text
    inputs = tokenizer.encode(article_text, return_tensors="pt", truncation=True)

    # Generate a summary id
    summary_ids = model.generate(inputs)

    # Decode the summary id to text
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary_text

# test_function_code --------------------

def test_summarize_article():
    print("Testing summarize_article function started.")

    # Test case: Summarizing a sample text
    sample_text = "Climate change is causing a rise in sea levels, resulting in the loss of coastal habitats and islands."
    expected_summary = "Climate change causes sea level rise and habitat loss."

    print("Test case [1/1] started.")
    actual_summary = summarize_article(sample_text)
    assert expected_summary in actual_summary, f"Test case [1/1] failed: Expected summary to contain '{{expected_summary}}', but got '{{actual_summary}}'."

    print("Testing summarize_article function finished.")

# Run the test function
test_summarize_article()