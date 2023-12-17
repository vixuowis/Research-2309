# requirements_file --------------------

!pip install -U torch transformers

# function_import --------------------

import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

# function_code --------------------

def get_jupiter_overview(text, question):
    '''
    Provides an overview on how Jupiter became the largest planet in our solar system by answering a given question.

    Args:
        text (str): A string containing informative text about Jupiter.
        question (str): A question regarding Jupiter's formation and growth.

    Returns:
        str: The answer to the given question based on the provided text.

    Raises:
        ValueError: If any of the inputs are not strings.

    '''
    if not all(isinstance(arg, str) for arg in [text, question]):
        raise ValueError('All inputs must be strings.')

    tokenizer = AutoTokenizer.from_pretrained('valhalla/longformer-base-4096-finetuned-squadv1')
    model = AutoModelForQuestionAnswering.from_pretrained('valhalla/longformer-base-4096-finetuned-squadv1')
    encoding = tokenizer(question, text, return_tensors='pt')
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']
    start_scores, end_scores = model(input_ids, attention_mask=attention_mask)
    all_tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    answer_tokens = all_tokens[torch.argmax(start_scores) :torch.argmax(end_scores)+1]
    answer = tokenizer.decode(tokenizer.convert_tokens_to_ids(answer_tokens))
    return answer

# test_function_code --------------------

def test_get_jupiter_overview():
    print("Testing started.")
    jupiter_text = 'Jupiter is the largest planet in our solar system due to its massive gas and dust accumulation during the planetary formation phase.'

    # Testing case 1: Valid input
    print("Testing case [1/2] started.")
    question1 = 'Why is Jupiter the largest planet?'
    assert get_jupiter_overview(jupiter_text, question1) == 'due to its massive gas and dust accumulation during the planetary formation phase', 'Test case [1/2] failed: Incorrect answer.'

    # Testing case 2: Invalid input
    print("Testing case [2/2] started.")
    try:
        get_jupiter_overview(123, 'Is Jupiter big?')
        assert False, 'Test case [2/2] failed: ValueError not raised for non-string input.'
    except ValueError:
        pass

    print("Testing finished.")

# call_test_function_line --------------------

test_get_jupiter_overview()