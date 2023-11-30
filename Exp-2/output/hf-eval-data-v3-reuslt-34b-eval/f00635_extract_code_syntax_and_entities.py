# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification

# function_code --------------------

def extract_code_syntax_and_entities(text):
    """
    Extracts code syntax and named entities from a text taken from StackOverflow.

    Args:
        text (str): The text from which to extract code syntax and named entities.

    Returns:
        dict: A dictionary containing the classified tokens and their corresponding labels.

    Raises:
        OSError: If there is an error in loading the pre-trained model or tokenizer.
    """    

    try:
        # load tokenizer and model from local disk if not cached
        tokenizer = AutoTokenizer.from_pretrained('./syntax_entity_code/tokenizer', do_lower_case=False)
        model = AutoModelForTokenClassification.from_pretrained('./syntax_entity_code/model')
    except OSError:
        raise Exception("Error loading syntax and entity code tokenizer or model!")

    # classify the input text
    tokens = tokenizer.encode(text)
    output = model(torch.tensor([tokens]))[0]
    predictions = torch.argmax(output, dim=2)
    
    return {i:j for i, j in zip(tokenizer.convert_ids_to_tokens(tokens), [ENTITY_CLASSES[prediction].split('-')[1] for prediction in predictions[0]])}

# test_function_code --------------------

def test_extract_code_syntax_and_entities():
    """
    Tests the function extract_code_syntax_and_entities.
    """
    test_text = 'How to use the AutoModelForTokenClassification from Hugging Face Transformers?'
    result = extract_code_syntax_and_entities(test_text)
    assert isinstance(result, dict), 'The result should be a dictionary.'
    assert 'tokens' in result, 'The result dictionary should have a key named tokens.'
    assert 'labels' in result, 'The result dictionary should have a key named labels.'
    assert isinstance(result['tokens'], list), 'The tokens should be a list.'
    assert isinstance(result['labels'], list), 'The labels should be a list.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_extract_code_syntax_and_entities()