# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import AutoModelWithLMHead, AutoTokenizer

# function_code --------------------

def gen_creative_sentence(words, max_length=32):
    """Generate a creative sentence based on given words using a pretrained model.

    Args:
        words (str): A string containing comma-separated words to base the sentence on.
        max_length (int, optional): The maximum length of the generated sentence. Defaults to 32.

    Returns:
        str: The generated creative sentence.

    Raises:
        ValueError: If 'words' is not provided or empty.
    """
    if not words:
        raise ValueError('The words parameter is required and cannot be empty.')

    tokenizer = AutoTokenizer.from_pretrained('mrm8488/t5-base-finetuned-common_gen')
    model = AutoModelWithLMHead.from_pretrained('mrm8488/t5-base-finetuned-common_gen')

    input_text = ' '.join(words.split(','))  # Convert comma-separated string to space-separated
    features = tokenizer([input_text], return_tensors='pt')
    output = model.generate(input_ids=features['input_ids'], attention_mask=features['attention_mask'], max_length=max_length)
    sentence = tokenizer.decode(output[0], skip_special_tokens=True)
    return sentence

# test_function_code --------------------

def test_gen_creative_sentence():
    print("Testing started.")

    # Test case 1: words provided
    print("Testing case [1/3] started.")
    sentence = gen_creative_sentence('tree,plant,ground,hole,dig')
    assert sentence and isinstance(sentence, str), f'Test case [1/3] failed: Expected a non-empty string.'

    # Test case 2: empty words
    print("Testing case [2/3] started.")
    try:
        gen_creative_sentence('')
    except ValueError as e:
        assert str(e) == 'The words parameter is required and cannot be empty.', f'Test case [2/3] failed: {e}'

    # Test case 3: very long words string
    print("Testing case [3/3] started.")
    very_long_words = ','.join(['word']*10)
    sentence = gen_creative_sentence(very_long_words)
    assert len(sentence.split()) <= 32, f'Test case [3/3] failed: Generated sentence exceeds max_length.'

    print("Testing finished.")

# call_test_function_line --------------------

test_gen_creative_sentence()