from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_name = 'csebuetnlp/mT5_multilingual_XLSum'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def summarize_news_article(article_text):
    '''
    This function takes in a string of international news article text and returns a summarized version of the text.
    It uses the 'csebuetnlp/mT5_multilingual_XLSum' model from the Transformers library to perform the summarization.
    '''
    input_ids = tokenizer.encode(article_text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    output_ids = model.generate(input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return summary