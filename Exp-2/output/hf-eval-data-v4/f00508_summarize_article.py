# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def summarize_article(long_news_article):
    """
    Summarizes a long news article using a pre-trained IT5-based model from Hugging Face Transformes.
    
    Args:
    long_news_article (str): A string containing the long news article to be summarized.

    Returns:
    str: A short summary of the news article.
    """
    summarizer = pipeline('summarization', model='it5/it5-base-news-summarization')
    summary = summarizer(long_news_article)[0]['summary_text']
    return summary

# test_function_code --------------------

def test_summarize_article():
    print("Testing started.")
    sample_data = "Dal 31 maggio ... sulla loro varietà."

    # 测试用例 1: 检查summarize_article是否返回字符串
    print("Testing case [1/1] started.")
    summary = summarize_article(sample_data)
    assert isinstance(summary, str), f"Test case [1/1] failed: The summary should be a string."
    print("Testing finished.")

# 运行测试函数
test_summarize_article()