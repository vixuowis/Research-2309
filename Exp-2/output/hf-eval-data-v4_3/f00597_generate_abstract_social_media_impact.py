# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import T5Tokenizer, T5Model

# function_code --------------------

def generate_abstract_social_media_impact(input_text: str) -> str:
    """
    Generates an abstract summarizing the impacts of social media on mental health.

    Args:
        input_text (str): The input sentence to be summarized.

    Returns:
        str: The generated abstract summarizing the key findings.

    Raises:
        ValueError: If the input_text is empty.
    """
    if not input_text:
        raise ValueError('Input text should not be empty.')

    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    model = T5Model.from_pretrained('t5-large')
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    decoder_input_ids = tokenizer('summarize: ', return_tensors='pt').input_ids
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)
    last_hidden_states = outputs.last_hidden_state

    summary = tokenizer.decode(last_hidden_states.squeeze(), skip_special_tokens=True)
    return summary


# test_function_code --------------------

def test_generate_abstract_social_media_impact():
    print("Testing started.")

    # Example text to summarize
    example_text = "Studies have shown the impacts of social media on mental health"

    # Test case 1
    print("Testing case [1/1] started.")
    summary = generate_abstract_social_media_impact(example_text)
    assert isinstance(summary, str), f"Test case [1/1] failed: The result is not a string."
    print("Testing successfully completed.")


# call_test_function_line --------------------

test_generate_abstract_social_media_impact()