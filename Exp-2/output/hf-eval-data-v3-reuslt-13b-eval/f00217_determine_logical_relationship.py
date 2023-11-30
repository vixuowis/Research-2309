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
    # Load tokenizer and model from pre-trained checkpoint. 
    model_name = "albert-xxlarge-v2"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Tokenize and encode the two sentences. 
    inputs1 = tokenizer([text1], return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer([text2], return_tensors="pt", padding=True, truncation=True)
    
    # Move model and inputs to the device (CPU or GPU). 
    model.to(device)
    inputs1["input_ids"] = inputs1["input_ids"].to(device)
    inputs1["attention_mask"] = inputs1["attention_mask"].to(device)
    inputs2["input_ids"] = inputs2["input_ids"].to(device)
    inputs2["attention_mask"] = inputs2["attention_mask"].to(device)
    
    # Make a prediction. 
    model.eval()
    with torch.no_grad():
        outputs1 = model(**inputs1)
        outputs2 = model(**inputs2)
        
    return {
        "entailment": (outputs1[0][0] + outputs2[0][0]) / 2, 
        "contradiction": (outputs1[0][1] + outputs2[0][1]) / 2, 
        "neutral": (outputs1[0][2] + outputs2[0][2]) / 2
    }

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