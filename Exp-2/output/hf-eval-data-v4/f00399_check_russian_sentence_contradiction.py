# requirements_file --------------------

!pip install -U transformers==4.11.3 torch==1.9.0 sentencepiece==0.1.96

# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# function_code --------------------

def check_russian_sentence_contradiction(sentence1: str, sentence2: str) -> bool:
    """
    Check if one Russian sentence logically contradicts another.

    Args:
    sentence1 (str): The first Russian sentence.
    sentence2 (str): The second Russian sentence.

    Returns:
    bool: True if there is a contradiction, False otherwise.
    """
    model_checkpoint = 'cointegrated/rubert-base-cased-nli-threeway'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        model.cuda()

    with torch.inference_mode():
        encoding = tokenizer(sentence1, sentence2, return_tensors='pt').to(model.device)
        output = model(**encoding)
        probabilities = torch.softmax(output.logits, dim=-1)
        predicted_label_index = torch.argmax(probabilities, dim=-1)
        contradiction_index = model.config.label2id['contradiction']

        return predicted_label_index.item() == contradiction_index

# test_function_code --------------------

def test_check_russian_sentence_contradiction():
    print("Testing check_russian_sentence_contradiction function.")

    # TestCase 1: Sentences with known contradiction
    sentence1 = 'Москва является столицей России.'
    sentence2 = 'Питер - столица России.'
    assert check_russian_sentence_contradiction(sentence1, sentence2), "TestCase 1 failed: Sentences should contradict."

    # TestCase 2: Sentences that do not contradict
    sentence1 = 'Москва - большой город.'
    sentence2 = 'В России есть много городов.'
    assert not check_russian_sentence_contradiction(sentence1, sentence2), "TestCase 2 failed: Sentences should not contradict."

    print("All test cases passed!")