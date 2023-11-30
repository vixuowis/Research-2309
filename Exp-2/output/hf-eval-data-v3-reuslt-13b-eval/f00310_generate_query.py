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
    
    if type(document) != str or len(document) == 0:
        raise ValueError("Document must be a non-empty string.")
        
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)
    
    document = "summarize: " + document
    
    encoding = tokenizer(document, return_tensors="pt")
    input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]
        
    outputs = model.generate(
        input_ids=input_ids, 
        attention_mask=attention_masks,
        return_dict_in_generate=True,
        output_scores=True,
        num_beams=10,
        num_return_sequences=25,
    )
    
    results = []
    for i in range(len(outputs["sequences"])):
        
        query = tokenizer.decode(outputs["sequences"][i], skip_special_tokens=True)
            
        if len(query) > 0:
            results.append((query, outputs["scores"][i].item()))
    
    return sorted(results, key = lambda x : x[1], reverse=True)[0][0]

# function_call --------------------

with open('./data/documents/doc-1.txt', 'r') as file:
    document = file.read()
    
print(generate_query(document))

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