# requirements_file --------------------

!pip install -U transformers huggingface-hub pytorch

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def fill_legal_document_gap(text, masked_index):
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-small-uncased')
    model = AutoModel.from_pretrained('nlpaueb/legal-bert-small-uncased')

    # Tokenize the input text and convert to tensor
    inputs = tokenizer(text, return_tensors='pt')

    # Predict the masked word
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = outputs.logits

    # Get the predicted token and decode it
    predicted_index = torch.argmax(predictions[0, masked_index]).item()
    predicted_token = tokenizer.convert_ids_to_tokens(predicted_index)

    # Replace the masked token with the predicted token
    filled_text = text.replace(tokenizer.mask_token, predicted_token)
    return filled_text

# test_function_code --------------------

def test_fill_legal_document_gap():
    print("Testing fill_legal_document_gap function.")

    # Example legal text with a masked token
    text = "The party shall pay the full amount within 30 [MASK] days after receiving the invoice."
    masked_index = 9  # [MASK] token index in the text

    # Expected word to fill the gap
    expected_word = 'calendar'

    # Run the function to fill the gap
    filled_text = fill_legal_document_gap(text, masked_index)

    # Assert if the gap is filled with the correct word
    assert expected_word in filled_text, f"Test failed, expected word '{expected_word}' to fill the gap, but got '{filled_text}'."
    print("Test passed, gap filled correctly with word '{expected_word}'.")

# Run the test
test_fill_legal_document_gap()