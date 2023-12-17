# requirements_file --------------------

!pip install -U transformers==4.24.0 torch==1.12.1 datasets==2.6.1 tokenizers==0.13.2

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_summary_and_question(input_text: str) -> str:
    """
    Generates a summary and open-ended question from the provided text.

    Args:
        input_text (str): The speech-to-text converted input that needs to be summarized.

    Returns:
        str: A string containing a summary and an open-ended question.

    Raises:
        ValueError: If the input_text is not provided or empty.
    """
    if not input_text:
        raise ValueError("Input text must not be empty.")

    # Initialize the tokenizer and model from the pre-trained model
    tokenizer = AutoTokenizer.from_pretrained('Qiliang/bart-large-cnn-samsum-ChatGPT_v3')
    model = AutoModelForSeq2SeqLM.from_pretrained('Qiliang/bart-large-cnn-samsum-ChatGPT_v3')

    # Generate the tokenized input
    tokenized_input = tokenizer(input_text, return_tensors="pt")

    # Generate the summary and question
    outputs = model.generate(**tokenized_input)

    # Decode the generated ids to a string
    summary_and_question = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return summary_and_question

# test_function_code --------------------

def test_generate_summary_and_question():
    print("Testing started.")
    sample_data = "Sample speech to text conversion data that needs to be summarized and questioned."

    # Test case 1: Check if function returns a non-empty string
    print("Testing case [1/1] started.")
    result = generate_summary_and_question(sample_data)
    assert isinstance(result, str) and len(result) > 0, "Test case [1/1] failed: The function should return a non-empty string."
    print("Testing finished.")

# call_test_function_line --------------------

test_generate_summary_and_question()