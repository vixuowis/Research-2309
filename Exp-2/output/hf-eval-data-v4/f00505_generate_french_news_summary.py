# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import BarthezTokenizer, BarthezModel

# function_code --------------------

def generate_french_news_summary(news_article_french: str) -> str:
    """
    Summarize a French news article using the pre-trained BarthezModel

    :param news_article_french: The French news article to summarize
    :return: The summary of the news article
    """
    tokenizer = BarthezTokenizer.from_pretrained('moussaKam/barthez-orangesum-abstract')
    model = BarthezModel.from_pretrained('moussaKam/barthez-orangesum-abstract')

    inputs = tokenizer(news_article_french, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids=inputs["input_ids"])

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary

# test_function_code --------------------

def test_generate_french_news_summary():
    print("Testing started.")
    # Provide a French news article
    news_article_french = "Le président a annoncé une nouvelle politique économique face à la crise."

    # Expected summary (this is just an example as actual summary can differ)
    expected_summary = "Le président annonce une nouvelle politique économique."

    # Test case 1
    print("Testing case [1/1] started.")
    summary = generate_french_news_summary(news_article_french)
    assert summary, f"Test case [1/1] failed: Summary not generated"
    print("Summary:", summary)
    print("Expected Summary:", expected_summary)
    print("Testing finished.")

# Run the test function
test_generate_french_news_summary()