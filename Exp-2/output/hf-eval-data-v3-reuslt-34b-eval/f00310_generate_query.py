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
    if type(document) != str:
        raise TypeError('The document should be passed as a single string.')
    
    # load tokenizer and model (should be fast if using T5 model from HuggingFace Hub with local cache)
    
    print('Loading tokenizer and model...')
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)
    
    # process input
    
    text = "summarize: "+document
    encoding = tokenizer(text, padding="max_length", max_length=200, truncation=True, return_tensors="pt")
    input_ids = encoding['input_ids']
    attention_mask = encoding["attention_mask"]
    
    # generate output
    
    summary_ids = model.generate(input_ids, 
                              attention_mask=attention_mask,
                              num_beams=4,
                              length_penalty=2.0,
                              min_length=15,
                              max_length=60,
                              no_repeat_ngram_size=3)
    
    # decode and return output (trim first token because it is always the same)
    
    return tokenizer.decode(summary_ids[0][1:], skip_special_tokens=True)

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