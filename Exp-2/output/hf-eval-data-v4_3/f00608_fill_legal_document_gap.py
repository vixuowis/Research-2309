# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def fill_legal_document_gap(text, mask_token='[MASK]'):
    """
    Automatically fill in the gap of a legal document text.

    Args:
        text (str): The text of the legal document with a mask token indicating the gap to fill.
        mask_token (str): The token used to indicate the gap in the text. Default is '[MASK]'.

    Returns:
        str: The text of the legal document with the gap filled.

    Raises:
        ValueError: If mask_token is not found in the text.
    """
    if mask_token not in text:
        raise ValueError(f"Mask token '{mask_token}' not found in the text.")

    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-small-uncased')
    model = AutoModel.from_pretrained('nlpaueb/legal-bert-small-uncased')

    # Tokenize the text
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Predict the masked token
    with torch.no_grad():
        outputs = model(input_ids)
    logits = outputs.logits
    predicted_token_id = logits[0].argmax(axis=1)
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id)[0]

    # Replace the mask token with the predicted token
    filled_text = text.replace(mask_token, predicted_token)
    return filled_text

# test_function_code --------------------

def test_fill_legal_document_gap():
    print("Testing started.")

    # Test case 1: Fill a single gap
    print("Testing case [1/3] started.")
    sample_text_with_single_gap = 'The party shall deliver the [MASK] documents within 5 days.'
    filled_text = fill_legal_document_gap(sample_text_with_single_gap)
    assert '[MASK]' not in filled_text, f"Test case [1/3] failed: Mask still present in text: {filled_text}"

    # Test case 2: Handle no gap in the text
    print("Testing case [2/3] started.")
    sample_text_without_gap = 'The party shall deliver the necessary documents within 5 days.'
    try:
        _ = fill_legal_document_gap(sample_text_without_gap)
        assert False, "Test case [2/3] failed: ValueError not raised for text without mask."
    except ValueError:
        pass  # This is expected

    # Test case 3: Custom mask token
    print("Testing case [3/3] started.")
    custom_mask_token = '<mask>'
    sample_text_with_custom_gap = f'The party shall deliver the {custom_mask_token} documents within 5 days.'
    filled_text_with_custom_gap = fill_legal_document_gap(sample_text_with_custom_gap, mask_token=custom_mask_token)
    assert custom_mask_token not in filled_text_with_custom_gap, "Test case [3/3] failed: Custom mask token still present in text."
    print("Testing finished.")

# call_test_function_line --------------------

test_fill_legal_document_gap()