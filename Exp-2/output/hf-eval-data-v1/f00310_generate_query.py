from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the pre-trained T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')

def generate_query(document):
    """
    This function takes a long text document as input and generates a query using a pre-trained T5 model.
    The generated query can be used to search for the most relevant articles.
    
    Args:
    document (str): The long text document.
    
    Returns:
    str: The generated query.
    """
    # Encode the document into input tensors
    inputs = tokenizer.encode("generate query: " + document, return_tensors="pt", max_length=512, truncation=True)
    # Generate the query
    outputs = model.generate(inputs, num_return_sequences=1, max_length=40)
    # Decode the output tensors into a string query
    return tokenizer.decode(outputs[0], skip_special_tokens=True)