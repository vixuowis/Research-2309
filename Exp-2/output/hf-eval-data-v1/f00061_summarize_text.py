from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Function to summarize text using the mT5 multilingual XLSum model
# @param: article_text - The text to be summarized
# @return: summary - The summarized text

def summarize_text(article_text):
    model_name = 'csebuetnlp/mT5_multilingual_XLSum'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer.encode(article_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    output_ids = model.generate(input_ids, max_length=84, no_repeat_ngram_size=2, num_beams=4)
    summary = tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return summary