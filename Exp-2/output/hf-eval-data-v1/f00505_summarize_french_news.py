from transformers import BarthezTokenizer, BarthezModel


def summarize_french_news(news_article_french):
    """
    This function takes a French news article as input and returns a short summary of the article.
    It uses the Barthez model from Hugging Face Transformers, which has been pre-trained on the orangeSum dataset.
    
    Parameters:
    news_article_french (str): The French news article to be summarized.
    
    Returns:
    str: The summary of the news article.
    """
    tokenizer = BarthezTokenizer.from_pretrained('moussaKam/barthez-orangesum-abstract')
    model = BarthezModel.from_pretrained('moussaKam/barthez-orangesum-abstract')
    inputs = tokenizer(news_article_french, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(input_ids=inputs["input_ids"])
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary