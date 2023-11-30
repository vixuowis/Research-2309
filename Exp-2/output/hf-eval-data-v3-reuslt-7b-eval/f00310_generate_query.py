# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def generate_query(document):
    """
    Generate a query from a given document using a pre-trained T5 model.

    Args:
        document (str): The document from which to generate the query.

    Returns:
        str: The generated query.

    Raises:
        ValueError: If the document is not a string or is empty.
    """
    
    # Check if the argument is valid.
    if not isinstance(document, str) or len(document.strip()) == 0:
        raise ValueError('Document must be provided as a non-empty string!')
        
    # Load the pre-trained model and tokenizer.
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    t5model = T5ForConditionalGeneration.from_pretrained("rinna/japanese-gpt2-query-generation")
    
    # Encode the input document and query prompts.
    text = "generate questions: " + document
    encoding = tokenizer.encode_plus(text, return_tensors="pt", max_length=512, padding='max_length', truncation=True)
    
    # Generate the output using our pre-trained model.
    output = t5model.generate(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
    
    # Decode and return the generated query.
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    return decoded

# test_function_code --------------------

def test_generate_query():
    """
    Test the generate_query function.
    """
    # Test with a valid document
    document = 'This is a test document.'
    query = generate_query(document)
    assert isinstance(query, str), 'Query must be a string.'

    # Test with an empty document
    try:
        generate_query('')
    except ValueError as e:
        assert str(e) == 'Document must be a non-empty string.', 'Exception message must be correct.'

    # Test with a non-string document
    try:
        generate_query(None)
    except ValueError as e:
        assert str(e) == 'Document must be a non-empty string.', 'Exception message must be correct.'

    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_query()