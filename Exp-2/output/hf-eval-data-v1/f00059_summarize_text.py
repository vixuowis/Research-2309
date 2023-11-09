from transformers import PegasusForConditionalGeneration, PegasusTokenizer


def summarize_text(text):
    '''
    This function takes a text string as input and returns a summarized version of the text.
    It uses the PegasusForConditionalGeneration model from Hugging Face Transformers, which is pretrained on the CNN/DailyMail dataset.
    
    Args:
    text (str): The text to be summarized.
    
    Returns:
    str: The summarized text.
    '''
    model_name = 'google/pegasus-cnn_dailymail'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.encode(text, return_tensors='pt')
    summary_ids = model.generate(inputs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary