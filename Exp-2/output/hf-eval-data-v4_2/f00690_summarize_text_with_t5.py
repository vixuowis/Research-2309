# requirements_file --------------------

!pip install -U transformers

# function_import --------------------

from transformers import T5Tokenizer, T5Model

# function_code --------------------

def summarize_text_with_t5(text):
    """
    Summarizes the input text using the T5 base model.

    Args:
        text (str): The text to be summarized.

    Returns:
        str: The summarized text.

    Raises:
        RuntimeError: If there is an issue with model inference.
    """
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5Model.from_pretrained('t5-base')
    input_text = f"summarize: {text}"
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids
    decoder_input_ids = tokenizer("summarize:", return_tensors='pt').input_ids
    try:
        outputs = model.generate(input_ids, decoder_input_ids=decoder_input_ids)
        conclusion = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return conclusion
    except Exception as e:
        raise RuntimeError(f'T5 model inference failed: {e}')

# test_function_code --------------------

def test_summarize_text_with_t5():
    print("Testing started.")
    sample_text = "Owning a dog can help decrease stress levels, improve your mood, and increase physical activity."

    print("Testing case [1/1] started.")
    summary = summarize_text_with_t5(sample_text)
    assert type(summary) == str, f"Test case [1/1] failed: The function should return a string."
    print("Testing case [1/1] passed.")
    print("Testing finished.")

# call_test_function_line --------------------

test_summarize_text_with_t5()