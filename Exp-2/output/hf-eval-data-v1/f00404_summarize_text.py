from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Function to summarize a given text using PegasusForConditionalGeneration model
# from Hugging Face Transformers

def summarize_text(article_text):
    '''
    This function takes a long article text as input and returns a summarized version of the text.
    It uses the PegasusForConditionalGeneration model from Hugging Face Transformers.
    '''
    # Initialize the tokenizer and the model
    tokenizer = PegasusTokenizer.from_pretrained('tuner007/pegasus_summarizer')
    model = PegasusForConditionalGeneration.from_pretrained('tuner007/pegasus_summarizer')

    # Tokenize the input text
    inputs = tokenizer.encode(article_text, return_tensors='pt', truncation=True)

    # Generate the summary
    summary_ids = model.generate(inputs)

    # Decode the generated summary
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary_text