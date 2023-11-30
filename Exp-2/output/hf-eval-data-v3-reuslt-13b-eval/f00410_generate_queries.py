# function_import --------------------

from transformers import T5Tokenizer, T5ForConditionalGeneration

# function_code --------------------

def generate_queries(document: str) -> str:
    '''
    Generate possible user queries for a given document using a pre-trained T5 model.

    Args:
        document (str): The input document for which to generate queries.

    Returns:
        str: The generated queries.

    Raises:
        ValueError: If the input document is not a string.
    '''
    
    # Ensure that the argument is of type 'string'.
    if not isinstance(document, str):
        raise TypeError("The provided document must be a string.")
        
    # Load the model and tokenizer.
    t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    
    # Tokenize inputs, and padding and make them both an integer sequences. 
    # Note that the input document must be a string of sentences seperated by full stops (".").
    tokens = tokenizer([document], return_tensors="pt", padding=True)
    
    # Generate queries.
    query = t5_model.generate(tokens['input_ids'], 
                              num_beams=6, 
                              early_stopping=True, 
                              max_length=30,
                              repetition_penalty=2., 
                              length_penalty=.15)
    
    # Decode the tokenized queries.
    decoded = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in query]
    
    return ' '.join([sentence.lstrip(' ') for sentence in decoded])

# test_function_code --------------------

def test_generate_queries():
    '''
    Test the generate_queries function.
    '''
    document1 = 'This is a test document.'
    document2 = 'Another test document.'
    document3 = 123

    assert isinstance(generate_queries(document1), str)
    assert isinstance(generate_queries(document2), str)
    try:
        generate_queries(document3)
    except ValueError:
        pass
    else:
        raise AssertionError('ValueError not raised for non-string input.')

    print('All Tests Passed')


# call_test_function_code --------------------

test_generate_queries()