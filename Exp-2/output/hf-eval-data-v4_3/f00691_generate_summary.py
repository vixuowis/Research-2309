# requirements_file --------------------

import subprocess

requirements = ["transformers"]

for package in requirements:
    subprocess.run(['pip', 'install', '-U', package])

# function_import --------------------

from transformers import T5Tokenizer, T5Model

# function_code --------------------

def generate_summary(article_text):
    """
    Generate a summary for a lengthy article text using T5 large model.

    Args:
        article_text (str): The article text to summarize.

    Returns:
        str: The summarized text.

    Raises:
        ValueError: If the article_text is empty.
    """
    if not article_text:
        raise ValueError('The article text must not be empty.')

    tokenizer = T5Tokenizer.from_pretrained('t5-large')
    model = T5Model.from_pretrained('t5-large')

    input_ids = tokenizer("summarize: " + article_text, return_tensors='pt').input_ids
    decoder_input_ids = tokenizer("summarize: ", return_tensors='pt').input_ids
    outputs = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids)

    summarized_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summarized_text

# test_function_code --------------------

def test_generate_summary():
    print("Testing started.")
    
    sample_data = "Studies have shown that owning a dog is good for you."

    print("Testing case [1/1] started.")
    summarized_text = generate_summary(sample_data)
    assert summarized_text, f"Test case [1/1] failed: The summary should not be empty."
    
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_summary()