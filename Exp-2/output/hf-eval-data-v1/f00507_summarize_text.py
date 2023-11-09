from transformers import pipeline

# Function to summarize text using PEGASUS model
# Input: Text to be summarized
# Output: Summarized text

def summarize_text(text):
    # Create an instance of the PEGASUS summarizer
    summarizer = pipeline('summarization', model='google/pegasus-large')
    # Use the summarizer instance on the input text
    summary = summarizer(text)
    # Return the summarized text
    return summary[0]['summary_text']