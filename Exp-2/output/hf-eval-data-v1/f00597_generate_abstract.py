from transformers import T5Tokenizer, T5Model

def generate_abstract(input_text: str, decoder_text: str = 'summarize: '):
    """
    This function uses the pre-trained T5-large model from Hugging Face Transformers to generate an abstract summarizing key findings about the impacts of social media on mental health.
    
    Args:
    input_text (str): The input text to be summarized.
    decoder_text (str): The decoder input text. Default is 'summarize: '.
    
    Returns:
    str: The generated abstract.
    """
    # Load the pre-trained T5-large model and its corresponding tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    model = T5Model.from_pretrained('t5-large')
    
    # Use the tokenizer to encode input text and decoder input text
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    decoder_input_ids = tokenizer(decoder_text, return_tensors='pt').input_ids
    
    # Run the pre-trained T5-large model with encoded input and decoder input text
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    last_hidden_states = outputs.last_hidden_state
    
    # Decode the last hidden state to get the generated abstract
    summary = tokenizer.decode(last_hidden_states[0], skip_special_tokens=True)
    
    return summary