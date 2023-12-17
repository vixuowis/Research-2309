# requirements_file --------------------

!pip install -U torch transformers sentencepiece

# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# function_code --------------------

def detect_russian_sentence_contradiction(sentence1: str, sentence2: str) -> bool:
    """Detect if one Russian sentence contradicts another.

    Args:
        sentence1: A string representing the first Russian sentence.
        sentence2: A string representing the second Russian sentence.
    
    Returns:
        A boolean indicating whether a contradiction is detected between the two sentences.

    Raises:
        ValueError: If any of the sentences are empty or not provided.
    """
    if not sentence1 or not sentence2:
        raise ValueError('Both sentences must be provided and non-empty.')
    model_checkpoint = 'cointegrated/rubert-base-cased-nli-threeway'
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        model.cuda()
    with torch.inference_mode():
        out = model(**tokenizer(sentence1, sentence2, return_tensors='pt').to(model.device))
        proba = torch.softmax(out.logits, -1).cpu().numpy()[0]
    predicted_label = {v: k for k, v in model.config.id2label.items()}[proba.argmax()]
    return predicted_label == 'contradiction'

# test_function_code --------------------

def test_detect_russian_sentence_contradiction():
    print('Testing started.')
    # Assuming sentences with clear logical relationships for simplicity
    test_cases = [
        ('', 'sentence', False, 'Empty first sentence'),
        ('sentence', '', False, 'Empty second sentence'),
        ('Противоречащее предложение', 'Логически несвязанное', False, 'Related sentences without contradiction'),
        ('Он пошел в школу', 'Он остался дома', True, 'Direct contradiction')
    ]
    for i, (sentence1, sentence2, expected, desc) in enumerate(test_cases, 1):
        print(f'Testing case [{i}/{len(test_cases)}] started: {desc}')
        result = detect_russian_sentence_contradiction(sentence1, sentence2)
        assert result == expected, f'Test case [{i}/{len(test_cases)}] failed: {desc}'
    print('Testing finished.')

# call_test_function_line --------------------

test_detect_russian_sentence_contradiction()