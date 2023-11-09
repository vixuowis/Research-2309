from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def generate_summary_and_question(input_text):
    """
    This function takes in a string of text and returns a summary and open-ended question.
    It uses the Hugging Face Transformers library and a pre-trained model for text summarization and open-ended question generation.
    
    Args:
    input_text (str): The text to be summarized and turned into an open-ended question.
    
    Returns:
    str: The generated summary and open-ended question.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('Qiliang/bart-large-cnn-samsum-ChatGPT_v3')
    model = AutoModelForSeq2SeqLM.from_pretrained('Qiliang/bart-large-cnn-samsum-ChatGPT_v3')
    
    # Tokenize the input text
    tokenized_input = tokenizer(input_text, return_tensors="pt")
    
    # Generate the summary and question
    summary_and_question = model.generate(**tokenized_input)
    
    return summary_and_question