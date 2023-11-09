from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def summarize_news_article(article_text):
    '''
    This function takes a news article as input and returns a summarized version of the article.
    It uses the 'csebuetnlp/mT5_multilingual_XLSum' model from the Transformers library to perform the summarization.
    '''
    model_name = 'csebuetnlp/mT5_multilingual_XLSum'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer(article_text, return_tensors='pt', padding=True, truncation=True, max_length=512).input_ids
    output_ids = model.generate(input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)[0]
    summary = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return summary