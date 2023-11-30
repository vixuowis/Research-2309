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
    
    # Load pre-trained model/tokenizer --------------------
    
    try:
        
        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained("mrm8488/spanbert-finetuned-squadv2", use_fast=True)
        
        # Load Model
        model = AutoModelForTokenClassification.from_pretrained(
            "mrm8488/spanbert-finetuned-stackoverlow", 
            num_labels=3, 
            id2label={0: 'CODE', 1: 'SYNTAX', 2:'NER'}, 
        )
        
    except OSError as e:
        
        print(e)

    # Tokenize text --------------------
    
    tokens = tokenizer.tokenize(text)
    
    # Classify tokens --------------------
    
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model(inputs)[0]
    
    # Create a dictionary containing the classified tokens and their corresponding labels --------------------
    
    classified_tokens = [
        {
            'token': tokenizer.convert_ids_to_tokens(token_id), 
            'label': outputs[0, idx].argmax(-1).item(), 
        }
        for idx, token_id in enumerate(inputs[0]) if tokenizer.convert_ids_to_tokens(token_id) != " "
    ]
    
    return classified_tokens

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