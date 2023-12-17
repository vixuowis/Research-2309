# requirements_file --------------------

!pip install -U transformers sentencepiece

# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# function_code --------------------

def predict_logical_relationship(text1: str, text2: str) -> dict:
    """
    Determine the logical relationship between two given sentences.

    Args:
        text1: A string representing the first sentence.
        text2: A string representing the second sentence.

    Returns:
        A dictionary with the probabilities of each logical relationship: entailment, contradiction, or neutral.

    Raises:
        ValueError: If the inputs are not strings.
    """
    if not isinstance(text1, str) or not isinstance(text2, str):
        raise ValueError('Both text1 and text2 must be strings.')

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

def test_predict_logical_relationship():
    print("Testing started.")
    # Test cases using example sentences
    
    # Test case 1: Entailment
    print("Testing case [1/3] started.")
    result = predict_logical_relationship("A soccer game with multiple males playing.", "Some men are playing a sport.")
    assert max(result, key=result.get) == 'entailment', f"Test case [1/3] failed: {result}"

    # Test case 2: Contradiction
    print("Testing case [2/3] started.")
    result = predict_logical_relationship("A soccer game with multiple males playing.", "The men are sleeping.")
    assert max(result, key=result.get) == 'contradiction', f"Test case [2/3] failed: {result}"

    # Test case 3: Neutral
    print("Testing case [3/3] started.")
    result = predict_logical_relationship("A soccer game with multiple males playing.", "The game is tied 0-0.")
    assert max(result, key=result.get) == 'neutral', f"Test case [3/3] failed: {result}"
    print("Testing finished.")

# call_test_function_line --------------------

test_predict_logical_relationship()