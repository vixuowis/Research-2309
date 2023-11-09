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
    model_checkpoint = 'cointegrated/rubert-base-cased-nli-threeway'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        model.cuda()

    with torch.inference_mode():
        out = model(**tokenizer(text1, text2, return_tensors='pt').to(model.device))
        proba = torch.softmax(out.logits, -1).cpu().numpy()[0]

    result = {v: proba[k] for k, v in model.config.id2label.items()}
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
    assert 'entailment' in result
    assert 'contradiction' in result
    assert 'neutral' in result

# call_test_function_code --------------------

test_determine_logical_relationship()