# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# function_code --------------------

def detect_russian_sentence_contradiction(sentence1: str, sentence2: str) -> bool:
    """
    Determine if one Russian sentence logically contradicts the information provided by another Russian sentence.

    Args:
        sentence1 (str): The first Russian sentence.
        sentence2 (str): The second Russian sentence.

    Returns:
        bool: True if contradiction is detected, False otherwise.
    """
    tokenizer = AutoTokenizer.from_pretrained("cointegrated/rutbert-tiny-rucontr")
    model = AutoModelForSequenceClassification.from_pretrained("cointegrated/rutbert-tiny-rucontr")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input1 = tokenizer(sentence1, return_tensors='pt').to(device)
    input2 = tokenizer(sentence2, return_tensors='pt').to(device)

    with torch.no_grad():
        output1 = model(**input1).logits
        output2 = model(**input2).logits
    
    return (output1 >= 0.5).item() != (output2 >= 0.5).item()


# test_function_code --------------------

def test_detect_russian_sentence_contradiction():
    assert detect_russian_sentence_contradiction('Это красная машина', 'Это синяя машина') == True
    assert detect_russian_sentence_contradiction('Это красная машина', 'Это красная машина') == False
    assert detect_russian_sentence_contradiction('Он любит кошек', 'Он ненавидит кошек') == True
    return 'All Tests Passed'


# call_test_function_code --------------------

test_detect_russian_sentence_contradiction()