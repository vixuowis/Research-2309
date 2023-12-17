# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch

# function_code --------------------

def predict_fill_in_the_blank(text, model_name='bert-base-chinese'):
    """
    Predicts the missing parts in a fill-in-the-blank text using a pre-trained BERT model.

    Args:
        text (str): The input text with one or more '[MASK]' tokens indicating the blanks.
        model_name (str): The name of the pre-trained BERT model to use.

    Returns:
        list: Predicted words for the blanks.
    """
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name)

    # Encode the text
    input_ids = tokenizer.encode(text, return_tensors='pt')

    # Predict the masked (blank) values
    with torch.no_grad():
        outputs = model(input_ids)
        predictions = outputs[0]

    # Get the predicted token ids
    predicted_token_ids = torch.argmax(predictions, dim=2)

    # Convert token ids to words
    predicted_tokens = [tokenizer.decode(token_id) for token_id in predicted_token_ids[0].tolist()]

    # Replace '[MASK]' with predicted tokens
    prediction_text = text
    for token in predicted_tokens:
        prediction_text = prediction_text.replace('[MASK]', token, 1)

    return predicted_tokens

# test_function_code --------------------

def test_predict_fill_in_the_blank():
    print("Testing predict_fill_in_the_blank function.")

    # Test case 1
    print("Test case [1/3] started.")
    text_with_mask = "我爱[MASK]。"
    predicted = predict_fill_in_the_blank(text_with_mask)
    assert len(predicted) > 0, f"Test case [1/3] failed: Expected at least one prediction, got {predicted}"

    # Test case 2
    print("Test case [2/3] started.")
    text_with_masks = "今天[MASK]很[MASK]。"
    predicted = predict_fill_in_the_blank(text_with_masks)
    assert len(predicted) == 2, f"Test case [2/3] failed: Expected two predictions, got {len(predicted)}"

    # Test case 3
    print("Test case [3/3] started.")
    text_no_mask = "我爱编程。"
    predicted = predict_fill_in_the_blank(text_no_mask)
    assert len(predicted) == 0, f"Test case [3/3] failed: Expected no predictions for text without masks, got {predicted}"

    print("All test cases passed.")