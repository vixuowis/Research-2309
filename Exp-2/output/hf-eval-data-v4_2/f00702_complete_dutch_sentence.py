# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoTokenizer, AutoModel

# function_code --------------------

def complete_dutch_sentence(input_text: str) -> str:
    """
    Complete the missing word in a Dutch sentence using BERTje, a pre-trained BERT model.

    Args:
        input_text (str): A Dutch sentence with a missing word represented as "___".

    Returns:
        str: The Dutch sentence with the missing word filled in.

    Raises:
        ValueError: If the input text does not contain the placeholder "___".
    """
    if '___' not in input_text:
        raise ValueError('The input text must contain the placeholder "___".')

    tokenizer = AutoTokenizer.from_pretrained('GroNLP/bert-base-dutch-cased')
    model = AutoModel.from_pretrained('GroNLP/bert-base-dutch-cased')
    input_text_with_mask = input_text.replace('___', tokenizer.mask_token)
    input_tokens = tokenizer.encode(input_text_with_mask, return_tensors='pt')
    mask_position = input_tokens.tolist()[0].index(tokenizer.mask_token_id)
    output = model(input_tokens)
    prediction = output.logits.argmax(dim=2)[0, mask_position].item()
    predicted_word = tokenizer.decode([prediction]).strip()
    return input_text.replace('___', predicted_word)

# test_function_code --------------------

from transformers import AutoTokenizer, AutoModel

def test_complete_dutch_sentence():
    print('Testing started.')
    # Test case 1: Placeholder in the sentence
    print('Testing case [1/3] started.')
    sentence_with_missing_word = 'Het is vandaag erg koud, dus vergeet niet je ___ mee te nemen.'
    completed_sentence = complete_dutch_sentence(sentence_with_missing_word)
    assert '___' not in completed_sentence, f'Test case [1/3] failed: Placeholder still present in the completed sentence.'

    # Test case 2: No placeholder in the sentence
    print('Testing case [2/3] started.')
    sentence_without_missing_word = 'Het is vandaag erg koud, dus vergeet niet je jas mee te nemen.'
    try:
        complete_dutch_sentence(sentence_without_missing_word)
        assert False, 'Test case [2/3] failed: Missing ValueError for missing placeholder.'
    except ValueError:
        pass  # Expected behavior

    # Test case 3: Placeholder is first word
    print('Testing case [3/3] started.')
    sentence_with_missing_first_word = '___ is vandaag erg koud, dus vergeet niet je jas mee te nemen.'
    completed_sentence_first_word = complete_dutch_sentence(sentence_with_missing_first_word)
    assert '___' not in completed_sentence_first_word, f'Test case [3/3] failed: Placeholder still present in the completed sentence with missing first word.'
    print('Testing finished.')

# Run the test function
test_complete_dutch_sentence()

# call_test_function_line --------------------

test_complete_dutch_sentence()