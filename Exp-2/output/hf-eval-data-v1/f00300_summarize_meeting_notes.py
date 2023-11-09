from transformers import PegasusForConditionalGeneration, PegasusTokenizer


def summarize_meeting_notes(meeting_notes):
    """
    This function takes in meeting notes as input and returns a summarized version of the notes.
    It uses the PegasusForConditionalGeneration model from the Hugging Face Transformers library,
    which is pretrained on the CNN/DailyMail dataset for abstractive summarization tasks.
    """
    model_name = 'google/pegasus-cnn_dailymail'
    tokenizer = PegasusTokenizer.from_pretrained(model_name)
    model = PegasusForConditionalGeneration.from_pretrained(model_name)
    inputs = tokenizer.encode(meeting_notes, return_tensors='pt')
    summary_ids = model.generate(inputs)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary