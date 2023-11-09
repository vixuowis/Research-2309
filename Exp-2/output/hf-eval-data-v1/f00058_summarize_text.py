from transformers import pipeline

# Function to summarize text using the PEGASUS model from Hugging Face Transformers
# The function takes a string of text as input and returns a summarized version of the text
# The PEGASUS model is a pre-trained model for abstractive summarization, developed by Google
# It is based on the Transformer architecture and trained on both C4 and HugeNews datasets
# The model is designed to extract gap sentences and generate summaries by stochastically sampling important sentences

def summarize_text(text):
    # Initialize the summarization pipeline with the PEGASUS model
    summarizer = pipeline('summarization', model='google/pegasus-xsum')
    # Use the pipeline to summarize the input text
    summary = summarizer(text)
    # Return the summarized text
    return summary[0]['summary_text']