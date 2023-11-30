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
    tokenizer = AutoTokenizer.from_pretrained("sberbank-ai/rugpt3large")
    model = AutoModelForSequenceClassification.from_pretrained("sberbank-ai/rugpt3large")

    input1 = tokenizer(sentence1, truncation=True)
    input2 = tokenizer(sentence2, truncation=True)
    
    output1 = model(**input1)[0].tolist()[0][-1] # get final logit value for contradiction
    output2 = model(**input2)[0].tolist()[0][-1] # get final logit value for contradiction

    return bool((output1 < output2) & (output1 > -1 * output2))
    

# test_function_code --------------------

def test_detect_russian_sentence_contradiction():
    assert detect_russian_sentence_contradiction('Это красная машина', 'Это синяя машина') == True
    assert detect_russian_sentence_contradiction('Это красная машина', 'Это красная машина') == False
    assert detect_russian_sentence_contradiction('Он любит кошек', 'Он ненавидит кошек') == True
    return 'All Tests Passed'


# call_test_function_code --------------------

test_detect_russian_sentence_contradiction()