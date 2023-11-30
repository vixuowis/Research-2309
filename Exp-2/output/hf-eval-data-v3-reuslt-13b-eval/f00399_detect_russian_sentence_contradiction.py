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
    
    # load pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepmipt/tardigrade-russian-bert-base-finetuned-paraphrase")
    model = AutoModelForSequenceClassification.from_pretrained("deepmipt/tardigrade-russian-bert-base-finetuned-paraphrase")

    # tokenize input texts
    text1 = [sentence1]
    text2 = [sentence2]
    encoded_text1 = tokenizer(text1, padding=True, truncation=True, max_length=512, return_tensors="pt")
    encoded_text2 = tokenizer(text2, padding=True, truncation=True, max_length=512, return_tensors="pt")

    # calculate embeddings for input texts and make predictions on them
    with torch.no_grad():    
        model.eval()
        outputs1 = model(encoded_text1["input_ids"], encoded_text1["attention_mask"])
        logits1 = outputs1.logits
        prediction1 = torch.argmax(logits1).numpy().tolist()
    
    with torch.no_grad():    
        model.eval()
        outputs2 = model(encoded_text2["input_ids"], encoded_text2["attention_mask"])
        logits2 = outputs2.logits
        prediction2 = torch.argmax(logits2).numpy().tolist()
    
    # check for contradiction between input sentences
    if (prediction1 != prediction2):
        return True
    else:
        return False

# test_function_code --------------------

def test_detect_russian_sentence_contradiction():
    assert detect_russian_sentence_contradiction('Это красная машина', 'Это синяя машина') == True
    assert detect_russian_sentence_contradiction('Это красная машина', 'Это красная машина') == False
    assert detect_russian_sentence_contradiction('Он любит кошек', 'Он ненавидит кошек') == True
    return 'All Tests Passed'


# call_test_function_code --------------------

test_detect_russian_sentence_contradiction()