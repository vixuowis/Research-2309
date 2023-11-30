# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# function_code --------------------

def determine_logical_relationship(text1: str, text2: str) -> dict:
    """
    Determine the logical relationship between two given sentences.

    Args:
        text1 (str): The first sentence.
        text2 (str): The second sentence.

    Returns:
        dict: A dictionary containing the probabilities of each logical relationship (entailment, contradiction, or neutral).
    """
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-portuguese-cased-sentiment")
    model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-portuguese-cased-sentiment")
    
    # Tokenize inputs
    tokens1 = tokenizer(text1, return_tensors="pt", truncation=True, padding='max_length') 
    tokens2 = tokenizer(text2, return_tensors="pt", truncation=True, padding='max_length') 
    
    # Perform inference
    outputs = model(torch.cat((tokens1["input_ids"], tokens2["input_ids"]), dim=-1))[0]
    predictions = torch.argmax(outputs, axis=1)
    
    result = {
        'entailment': float(predictions[0]),
        'neutral': float(predictions[1]),
        'contradiction': float(predictions[2])
    }
    
    return result

# test_function_code --------------------

def test_determine_logical_relationship():
    """
    Test the function determine_logical_relationship.
    """
    text1 = 'The cat is on the mat.'
    text2 = 'There is a cat on the mat.'
    result = determine_logical_relationship(text1, text2)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'entailment', 'contradiction', 'neutral'}

    text1 = 'It is raining.'
    text2 = 'The weather is sunny.'
    result = determine_logical_relationship(text1, text2)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'entailment', 'contradiction', 'neutral'}

    text1 = 'He is a boy.'
    text2 = 'She is a girl.'
    result = determine_logical_relationship(text1, text2)
    assert isinstance(result, dict)
    assert set(result.keys()) == {'entailment', 'contradiction', 'neutral'}

    return 'All Tests Passed'


# call_test_function_code --------------------

print(test_determine_logical_relationship())