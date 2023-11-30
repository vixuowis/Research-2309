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
        # load tokenizer and model for token classification
        tokenizer = AutoTokenizer.from_pretrained("dennlinger/codebert")
        model = AutoModelForTokenClassification.from_pretrained("dennlinger/codebert", return_dict=True)
    except OSError:
        raise OSError('There is an error in loading the pre-trained model or tokenizer')
    
    # classify tokens and extract label for each one
    input = tokenizer(text, add_special_tokens=False, return_tensors='pt', truncation=True)
    output = model(**input)
    labels = [model.config.id2label[prediction] for prediction in output['logits'].argmax(-1)[0]]
    
    # classify tokens and extract label for each one
    token_classification = dict()
    index = 0
    for token, label in zip(tokenizer.convert_ids_to_tokens(input['input_ids'][0]), labels):
        if token != 'Ġ':
            # add to dictionary
            token_classification[str(index)] = {'token': str(token), 'label': label}
            
            # increment index
            index += 1
    
    return token_classification

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