# requirements_file --------------------

!pip install -U transformers sentencepiece

# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# function_code --------------------

def determine_logical_relationship(text1, text2):
    """
    Determine the logical relationship between two sentences.

    Args:
        text1 (str): The first sentence.
        text2 (str): The second sentence.

    Returns:
        dict: A dictionary containing the probabilities of each logical relationship
              (entailment, contradiction, or neutral).
    """
    model_checkpoint = 'cointegrated/rubert-base-cased-nli-threeway'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        model.cuda()

    with torch.inference_mode():
        encoded_input = tokenizer(text1, text2, return_tensors='pt').to(model.device)
        output = model(**encoded_input)
        probabilities = torch.softmax(output.logits, -1).cpu().numpy()[0]

    result = {v: probabilities[k] for k, v in model.config.id2label.items()}
    return result

# test_function_code --------------------

def test_determine_logical_relationship():
    print("Testing determine_logical_relationship function.")

    # Example sentences
    text1 = "A dog is running."
    text2 = "An animal is moving."
    result = determine_logical_relationship(text1, text2)
    assert 'entailment' in result, "Test case failed: 'entailment' not in result"
    assert 'contradiction' in result, "Test case failed: 'contradiction' not in result"
    assert 'neutral' in result, "Test case failed: 'neutral' not in result"

    print("Testing finished successfully.")

# Execute the test
test_determine_logical_relationship()