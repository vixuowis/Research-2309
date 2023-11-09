from transformers import T5Tokenizer, T5ForConditionalGeneration


def generate_positive_review(book_summary):
    """
    This function converts a book summary into a positive book review using the T5-3B model from Hugging Face Transformers.
    
    Parameters:
    book_summary (str): The summary of the book.
    
    Returns:
    str: The positive book review.
    """
    # Load the T5-3B model
    model = T5ForConditionalGeneration.from_pretrained('t5-3b')
    # Load the T5 tokenizer
    tokenizer = T5Tokenizer.from_pretrained('t5-3b')
    # Preprocess the book summary
    input_text = 'Write a positive review: ' + book_summary
    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors='pt')
    # Generate the positive book review
    outputs = model.generate(inputs)
    # Decode the output tokens to obtain the positive book review text
    positive_review = tokenizer.decode(outputs[0])
    return positive_review