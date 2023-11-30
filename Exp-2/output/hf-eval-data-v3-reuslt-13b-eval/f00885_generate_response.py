# function_import --------------------

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# function_code --------------------

def generate_response(instruction: str, knowledge: str, dialog: list) -> str:
    """
    Generate a response based on the instruction, knowledge, and dialog.

    Args:
        instruction (str): Instruction on how to respond.
        knowledge (str): Knowledge about the situation.
        dialog (list): List of dialogues.

    Returns:
        str: Generated response.
    """

    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")
    model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-summarize-news")

    # concatenate instruction, knowledge and dialog into a single string
    source = " ".join([instruction, knowledge] + dialog)
    # tokenize the source text
    input_ids = tokenizer.encode(source, return_tensors="pt")
    # summarize the source text
    summary_ids = model.generate(input_ids, max_length=2048, min_length=512, length_penalty=2., num_beams=10)
    # decode the generated ids to text and remove special tokens from output
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


# test_function_code --------------------

def test_generate_response():
    """
    Test the generate_response function.
    """
    instruction = 'How can I respond to a customer complaint about late delivery?'
    knowledge = 'The courier had external delays due to bad winter weather.'
    dialog = ['Customer: My package is late. What is going on?', 'Support: I apologize for the inconvenience. I will check what is happening with the package and get back to you.']
    response = generate_response(instruction, knowledge, dialog)
    assert isinstance(response, str), 'The response should be a string.'
    assert len(response) > 0, 'The response should not be empty.'
    return 'All Tests Passed'


# call_test_function_code --------------------

test_generate_response()