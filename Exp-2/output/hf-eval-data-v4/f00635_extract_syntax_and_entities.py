# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForTokenClassification

# function_code --------------------

def extract_syntax_and_entities(text):
    """
    This function uses a pre-trained BERTOverflow model to extract code syntax and named entities from a text taken from StackOverflow.

    :param text: str - The text input from StackOverflow.
    :return: dict - A dictionary with the tokens and their corresponding classified labels.
    """
    tokenizer = AutoTokenizer.from_pretrained('lanwuwei/BERTOverflow_stackoverflow_github')
    model = AutoModelForTokenClassification.from_pretrained('lanwuwei/BERTOverflow_stackoverflow_github')
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    predictions = outputs[0].argmax(axis=-1)
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    labels = [model.config.id2label[p.item()] for p in predictions[0]]
    return {'tokens': tokens, 'labels': labels}

# test_function_code --------------------

def test_extract_syntax_and_entities():
    print("Testing started.")
    sample_text = 'How to extract entities using BERTOverflow?'

    # Expected result should contain tokens and labels related to StackOverflow context
    expected_labels_part = ['O', 'O', 'O', 'O', 'B-METHOD', 'O', 'B-METHOD', 'O']

    # Test case
    print("Testing case started.")
    result = extract_syntax_and_entities(sample_text)
    assert all(label in result['labels'][:8] for label in expected_labels_part), "Test case failed: The extracted labels do not match the expected labels part."
    print("Testing finished.")

# Running the test function
test_extract_syntax_and_entities()