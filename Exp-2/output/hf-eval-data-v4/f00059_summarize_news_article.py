# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# function_code --------------------

def summarize_news_article(article_text):
    model_name = 'google/pegasus-cnn_dailymail'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.encode(article_text, return_tensors='pt')
    summary_ids = model.generate(inputs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


# test_function_code --------------------

def test_summarize_news_article():
    print("Testing summarize_news_article function.")
    # Test case: Summarize a sample news article
    sample_article = "A new study suggests that eating chocolate at least once a week..."
    expected_keywords = ['chocolate', 'cognition', 'study']
    summary = summarize_news_article(sample_article)
    assert all(word in summary for word in expected_keywords), "Test failed: Keywords missing in summary."
    print("Test passed: Keywords present in summary.")

# Run the test
print("Running test for summarize_news_article function...")
test_summarize_news_article()
