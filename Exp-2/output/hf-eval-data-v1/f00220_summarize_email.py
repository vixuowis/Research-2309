from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Function to summarize long emails
# Uses the PegasusForConditionalGeneration model from Hugging Face Transformers
# The model is pretrained on the 'tuner007/pegasus_summarizer' dataset
# The function takes a string (long email) as input and returns a summarized version of the email

def summarize_email(email_text):
    model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_summarizer')
    tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_summarizer')
    input_ids = tokenizer(email_text, return_tensors='pt').input_ids
    summary_ids = model.generate(input_ids)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text