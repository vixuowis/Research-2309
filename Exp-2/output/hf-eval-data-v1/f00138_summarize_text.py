from transformers import BartTokenizer, BartForConditionalGeneration

# Function to summarize text using the pre-trained model 'sshleifer/distilbart-cnn-12-6'
def summarize_text(input_text):
    """
    This function takes in a long text and returns a summarized version of it.
    It uses the pre-trained model 'sshleifer/distilbart-cnn-12-6' from the Hugging Face Transformers library.
    """
    # Load the pre-trained model and tokenizer
    model = BartForConditionalGeneration.from_pretrained('sshleifer/distilbart-cnn-12-6')
    tokenizer = BartTokenizer.from_pretrained('sshleifer/distilbart-cnn-12-6')

    # Tokenize the input text and pass it to the model
    inputs = tokenizer(input_text, return_tensors='pt')
    summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=50, early_stopping=True)

    # Decode the model's output to get the summary
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary_text