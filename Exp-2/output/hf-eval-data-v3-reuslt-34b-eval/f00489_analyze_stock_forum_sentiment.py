# function_import --------------------

from transformers import RobertaForSequenceClassification, RobertaTokenizer, pipeline
import pandas as pd

# function_code --------------------

def analyze_stock_forum_sentiment(forum_posts):
    """
    Analyze the sentiment of a stock forum using a pre-trained model.

    Args:
        forum_posts (pd.Series): A pandas Series of forum posts.

    Returns:
        list: A list of sentiment analysis results for each post.
    """

    # Load transformers tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment"
    )
    model = RobertaForSequenceClassification.from_pretrained(
        "cardiffnlp/twitter-roberta-base-sentiment"
    )

    # Build function that classifies sentiment
    def predict_sentiment(x):

        encoded = tokenizer.encode_plus(
            x,
            max_length=128,
            add_special_tokens=True,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].tolist()[0]
        attention_mask = encoded["attention_mask"].tolist()[0]

        output = model(torch.tensor([input_ids]), torch.tensor([attention_mask]))

        # Apply Softmax at dimension 1
        probs = functools.reduce(torch.nn.Softmax(dim=1), output.logits)

        result = {
            "negative": probs[0][0].item(),
            "neutral": probs[0][1].item(),
            "positive": probs[0][2].item(),
        }

        return result

    # Build sentiment analysis pipeline
    text_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    # Perform analysis for each forum post
    results = []
    for post in forum_posts:
        result = predict_sentiment(post)
        results.append(result)
    
    return results

# test_function_code --------------------

def test_analyze_stock_forum_sentiment():
    """
    Test the analyze_stock_forum_sentiment function.
    """
    forum_posts = pd.Series(["Stock X is going up!", "I'm selling my shares.", "Buy now before it's too late!"])
    results = analyze_stock_forum_sentiment(forum_posts)
    assert isinstance(results, list), 'The result should be a list.'
    assert len(results) == len(forum_posts), 'The length of the result should be equal to the length of the input.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_analyze_stock_forum_sentiment()