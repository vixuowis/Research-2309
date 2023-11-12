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
    model_checkpoint = 'cointegrated/rubert-base-cased-nli-threeway'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        model.cuda()
    with torch.inference_mode():
        out = model(**tokenizer(sentence1, sentence2, return_tensors='pt').to(model.device))
        proba = torch.softmax(out.logits, -1).cpu().numpy()[0]
    predicted_label = {v: proba[k] for k, v in model.config.id2label.items()}
    return predicted_label['contradiction'] > predicted_label['neutral'] and predicted_label['contradiction'] > predicted_label['entailment']

# test_function_code --------------------

def test_detect_russian_sentence_contradiction():
    assert detect_russian_sentence_contradiction('Это красная машина', 'Это синяя машина') == True
    assert detect_russian_sentence_contradiction('Это красная машина', 'Это красная машина') == False
    assert detect_russian_sentence_contradiction('Он любит кошек', 'Он ненавидит кошек') == True
    return 'All Tests Passed'

# call_test_function_code --------------------

test_detect_russian_sentence_contradiction()