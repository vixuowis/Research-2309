# function_import --------------------

from transformers import T5Tokenizer, T5Model

# function_code --------------------

def generate_summary(input_text: str, model_name: str = 't5-large') -> str:
    '''
    Generate a summary for the given input text using the specified T5 model.

    Args:
        input_text (str): The text to be summarized.
        model_name (str): The name of the T5 model to be used for summarization. Default is 't5-large'.

    Returns:
        str: The generated summary.

    Raises:
        ValueError: If the input text is not a string.
    '''
    if not isinstance(input_text, str):
        raise ValueError('Input text must be a string.')

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5Model.from_pretrained(model_name)

    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids)

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary

# test_function_code --------------------

def test_generate_summary():
    '''
    Test the generate_summary function.
    '''
    input_text = 'Studies have shown the impacts of social media on mental health.'
    expected_output = 'Studies indicate that social media has significant effects on mental health.'

    output = generate_summary(input_text)

    assert isinstance(output, str), 'Output is not a string.'
    assert output == expected_output, 'Output does not match expected output.'

    print('All tests passed.')

# call_test_function_code --------------------

test_generate_summary()