# requirements_file --------------------

!pip install -U transformers pytorch

# function_import --------------------

from transformers import pipeline

# function_code --------------------

def analyze_movie_review_sentiment(review_text):
    """
    Analyze the sentiment of a given movie review using a pretrained model.

    Parameters:
    review_text (str): The text of the movie review to be analyzed.

    Returns:
    str: The sentiment of the review, either 'positive' or 'negative'.
    """
    # Initialize the sentiment analysis model
    classifier = pipeline('sentiment-analysis', model='lvwerra/distilbert-imdb')
    
    # Analyze the sentiment of the review
    result = classifier(review_text)
    
    # Extract the sentiment label
    review_sentiment = result[0]['label']
    
    return review_sentiment

# test_function_code --------------------

def test_analyze_movie_review_sentiment():
    print("Testing started.")
    
    # 测试用例 1：正面评论
    positive_review = "I love this movie! It was fantastic and the acting was superb!"
    print("Testing case [1/3] started.")
    assert analyze_movie_review_sentiment(positive_review) == 'POSITIVE', "Test case [1/3] failed: Expected 'POSITIVE'."

    # 测试用例 2：负面评论
    negative_review = "This movie was terrible. The plot was boring and predictable."
    print("Testing case [2/3] started.")
    assert analyze_movie_review_sentiment(negative_review) == 'NEGATIVE', "Test case [2/3] failed: Expected 'NEGATIVE'."

    # 测试用例 3：中性评论
    neutral_review = "The movie was okay, not great but not bad either."
    print("Testing case [3/3] started.")
    # 由于模型主要用于正面和负面情绪分类，中性评论可能倾向于其中一种
    # 因此，这个测试可能需要根据实际模型的输出进行调整
    assert analyze_movie_review_sentiment(neutral_review) in ['POSITIVE', 'NEGATIVE'], "Test case [3/3] failed: Expected 'POSITIVE' or 'NEGATIVE'."
    
    print("Testing finished.")

# 运行测试函数
test_analyze_movie_review_sentiment()