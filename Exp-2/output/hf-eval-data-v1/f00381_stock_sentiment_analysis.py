from transformers import pipeline
import pandas as pd


def stock_sentiment_analysis(stock_comments):
    '''
    This function uses the Hugging Face Transformers library to load a pre-trained model
    'zhayunduo/roberta-base-stocktwits-finetuned' and its accompanying tokenizer 'RobertaTokenizer'.
    The model has been fine-tuned on sentiment classification for stock-related comments.
    It takes a list of stock comments as input and returns the sentiment towards each stock.
    '''
    classifier = pipeline('text-classification', model='zhayunduo/roberta-base-stocktwits-finetuned', tokenizer='RobertaTokenizer')
    sentiment_results = classifier(stock_comments.tolist())
    return sentiment_results