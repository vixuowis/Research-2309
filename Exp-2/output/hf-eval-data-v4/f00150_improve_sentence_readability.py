# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import DebertaTokenizer, DebertaModel

# function_code --------------------

def improve_sentence_readability(sentence: str) -> str:
    # Load pretrained Deberta Tokenizer and Model
    tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-v2-xlarge')
    model = DebertaModel.from_pretrained('microsoft/deberta-v2-xlarge')

    # Prepare the sentence with the masked token
    masked_sentence = sentence.replace('...', '[MASK]')
    tokenized_input = tokenizer(masked_sentence, return_tensors='pt')

    # Get model output
    with torch.no_grad():
        output = model(**tokenized_input)

    # Decode the predicted token and replace the masked part
    predicted_token_id = torch.argmax(output.logits, dim=-1)[0, tokenized_input['input_ids'][0] == tokenizer.mask_token_id]
    predicted_token = tokenizer.decode(predicted_token_id)
    improved_sentence = masked_sentence.replace('[MASK]', predicted_token.strip())

    return improved_sentence

# test_function_code --------------------

def test_improve_sentence_readability():
    print('Testing improve_sentence_readability function...')

    # Test with a sample sentence
    sample_sentence = 'The weather today is ... nice.'
    expected_word = 'very'  # Assuming 'very' is the predicted token for the masked part

    improved = improve_sentence_readability(sample_sentence)
    assert expected_word in improved, f'Improved sentence does not contain the expected word: {expected_word}'

    print('Test passed!')

# Run the test function
test_improve_sentence_readability()