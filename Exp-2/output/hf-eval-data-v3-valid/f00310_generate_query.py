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
    if not isinstance(document, str) or not document:
        raise ValueError('Document must be a non-empty string.')

    tokenizer = T5Tokenizer.from_pretrained('castorini/doc2query-t5-base-msmarco')
    model = T5ForConditionalGeneration.from_pretrained('castorini/doc2query-t5-base-msmarco')

    inputs = tokenizer.encode('generate query: ' + document, return_tensors='pt', max_length=512, truncation=True)
    outputs = model.generate(inputs, num_return_sequences=1, max_length=40)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

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